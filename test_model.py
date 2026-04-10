# ============================================================
# Hawk-Eye: Model Testing Script
# Test trained model on single video or entire test dataset
# ============================================================

import os
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = r"D:\B22IN082\Major Project\NEW\SCVD_converted\Test"
MODEL_PATH = "final_violence_detection_model.h5"  # Use the best model
CLASSES = ["Normal", "Violence", "Weaponized"]
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 224
SEQUENCE_LENGTH = 30

# ============================================================
# FRAME EXTRACTION (Same as training)
# ============================================================

def extract_frames(video_path):
    """Extract frames from video for prediction"""
    frames = []
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // SEQUENCE_LENGTH, 1)

    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == SEQUENCE_LENGTH:
        return np.array(frames, dtype=np.float32)
    else:
        return None

# ============================================================
# LOAD MODEL
# ============================================================

def rebuild_model_architecture():
    """Rebuild the model architecture to match training"""
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import (
        TimeDistributed, GlobalAveragePooling2D, LSTM, Dense, Dropout
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    
    # Build base model
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze most layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    # Build full model
    model = Sequential()
    model.add(TimeDistributed(
        base_model,
        input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)
    ))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(LSTM(128, return_sequences=False, dropout=0.3, 
                   recurrent_dropout=0.3, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(NUM_CLASSES, activation="softmax"))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    return model

def load_trained_model(model_path=MODEL_PATH):
    """Load the trained model with compatibility handling"""
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("Available model files:")
        for file in os.listdir("."):
            if file.endswith(".h5"):
                print(f"  - {file}")
        return None
    
    print(f"Loading model from: {model_path}")
    
    # Try normal loading first
    try:
        model = load_model(model_path, compile=False)
        from tensorflow.keras.optimizers import Adam
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )
        print("✓ Model loaded successfully!\n")
        return model
    except Exception as e:
        print(f"⚠ Compatibility issue detected: {type(e).__name__}")
        print("🔧 Rebuilding model architecture and loading weights...")
        
        try:
            # Rebuild model and load weights
            model = rebuild_model_architecture()
            model.load_weights(model_path)
            print("✓ Model loaded successfully with weights!\n")
            return model
        except Exception as e2:
            print(f"❌ Failed to load weights: {str(e2)}")
            print("\nTip: Ensure the model file matches the training architecture.")
            return None

# ============================================================
# SINGLE VIDEO PREDICTION
# ============================================================

def predict_single_video(model, video_path, display=True):
    """Predict class for a single video"""
    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at {video_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Testing Video: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    # Extract frames
    print("Extracting frames...")
    frames = extract_frames(video_path)
    
    if frames is None:
        print("❌ Error: Could not process video")
        return None
    
    # Make prediction
    frames_batch = np.expand_dims(frames, axis=0)
    predictions = model.predict(frames_batch, verbose=0)[0]
    
    class_index = np.argmax(predictions)
    predicted_class = CLASSES[class_index]
    confidence = predictions[class_index] * 100
    
    # Display results
    if display:
        print("\n📊 Prediction Results:")
        print(f"{'─'*60}")
        for i, class_name in enumerate(CLASSES):
            prob = predictions[i] * 100
            bar = "█" * int(prob / 2)
            marker = " ← PREDICTED" if i == class_index else ""
            print(f"{class_name:12} : {prob:5.2f}% {bar}{marker}")
        print(f"{'─'*60}")
        print(f"\n🎯 Final Prediction: {predicted_class} (Confidence: {confidence:.2f}%)")
    
    return predicted_class, predictions

# ============================================================
# BATCH TESTING ON TEST DATASET
# ============================================================

def test_on_dataset(model, test_dir=TEST_DIR):
    """Test model on entire test dataset"""
    print(f"\n{'='*60}")
    print("TESTING ON ENTIRE TEST DATASET")
    print(f"{'='*60}\n")
    
    y_true = []
    y_pred = []
    video_count = 0
    
    for label, class_name in enumerate(CLASSES):
        class_path = os.path.join(test_dir, class_name)
        print(f"Testing {class_name} videos...")
        
        for video_file in os.listdir(class_path):
            video_path = os.path.join(class_path, video_file)
            
            frames = extract_frames(video_path)
            if frames is not None:
                frames_batch = np.expand_dims(frames, axis=0)
                predictions = model.predict(frames_batch, verbose=0)[0]
                predicted_class = np.argmax(predictions)
                
                y_true.append(label)
                y_pred.append(predicted_class)
                video_count += 1
    
    print(f"\n✓ Tested {video_count} videos\n")
    return y_true, y_pred

# ============================================================
# EVALUATION METRICS
# ============================================================

def display_evaluation_metrics(y_true, y_pred):
    """Display comprehensive evaluation metrics"""
    print(f"\n{'='*60}")
    print("EVALUATION METRICS")
    print(f"{'='*60}\n")
    
    # Classification Report
    print("📈 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class accuracy
    print("\n📊 Per-Class Performance:")
    print(f"{'─'*60}")
    for i, class_name in enumerate(CLASSES):
        total = np.sum(cm[i])
        correct = cm[i][i]
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"{class_name:12} : {correct:3d}/{total:3d} correct ({accuracy:5.2f}%)")
    print(f"{'─'*60}")
    
    # Overall accuracy
    overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm) * 100
    print(f"\n🎯 Overall Accuracy: {overall_accuracy:.2f}%")
    
    return cm

# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

def plot_confusion_matrix(cm, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Count'}, square=True, linewidths=1)
    
    plt.title('Confusion Matrix - Hawk-Eye Threat Detection', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved as '{save_path}'")
    plt.show()

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test Hawk-Eye Threat Detection Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on entire test dataset
  python test_model.py --full-test
  
  # Test single video
  python test_model.py --video "path/to/video.mp4"
  
  # Use different model
  python test_model.py --model "hawk_eye_threat_detector_final.h5" --full-test
        """
    )
    
    parser.add_argument('--video', type=str, 
                       help='Path to single video file to test')
    parser.add_argument('--full-test', action='store_true',
                       help='Run full test on entire test dataset')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                       help=f'Path to model file (default: {MODEL_PATH})')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting confusion matrix')
    
    args = parser.parse_args()
    
    # Load model
    model = load_trained_model(args.model)
    if model is None:
        return
    
    # Test single video
    if args.video:
        predict_single_video(model, args.video)
    
    # Full test on dataset
    if args.full_test:
        y_true, y_pred = test_on_dataset(model)
        cm = display_evaluation_metrics(y_true, y_pred)
        
        if not args.no_plot:
            plot_confusion_matrix(cm)
    
    # If no arguments, show help
    if not args.video and not args.full_test:
        parser.print_help()
        print("\n" + "="*60)
        print("Quick Start:")
        print("="*60)
        print("1. Test single video:")
        print('   python test_model.py --video "Test/Violence/video.mp4"')
        print("\n2. Full evaluation:")
        print("   python test_model.py --full-test")
        print("="*60)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
