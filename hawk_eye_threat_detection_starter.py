# ============================================================
# Hawk-Eye: AI-Powered Threat Detection for CCTV Surveillance
# Complete Starter Training Script (CNN + LSTM)
# Dataset: SCVD (Normal / Violence / Weaponized)
# ============================================================

import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    TimeDistributed,
    GlobalAveragePooling2D,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ============================================================
# 1. DATASET PATHS  (EDIT IF NEEDED)
# ============================================================

TRAIN_DIR = r"D:\B22IN082\Major Project\NEW\SCVD_converted\Train"
TEST_DIR  = r"D:\B22IN082\Major Project\NEW\SCVD_converted\Test"

CLASSES = ["Normal", "Violence", "Weaponized"]
NUM_CLASSES = len(CLASSES)

# ============================================================
# 2. PARAMETERS
# ============================================================

IMG_SIZE = 224
SEQUENCE_LENGTH = 30
BATCH_SIZE = 4
EPOCHS = 50  # Increased but early stopping will prevent overfitting
PATIENCE = 5  # Stop if no improvement for 5 epochs

# ============================================================
# 3. FRAME EXTRACTION FUNCTION
# ============================================================

def extract_frames(video_path):
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
# 4. VIDEO DATA GENERATOR (MEMORY EFFICIENT)
# ============================================================

def get_video_paths(directory):
    """Get all video paths and labels without loading into memory"""
    video_paths = []
    labels = []

    for label, class_name in enumerate(CLASSES):
        class_path = os.path.join(directory, class_name)
        print(f"Scanning {class_name} videos...")

        for video_file in os.listdir(class_path):
            video_path = os.path.join(class_path, video_file)
            video_paths.append(video_path)
            labels.append(label)

    return video_paths, labels


class VideoDataGenerator(Sequence):
    """Custom data generator for loading videos in batches"""
    
    def __init__(self, video_paths, labels, batch_size, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.video_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = []
        batch_y = []
        
        for i in batch_indices:
            frames = extract_frames(self.video_paths[i])
            if frames is not None:
                batch_X.append(frames)
                batch_y.append(self.labels[i])
        
        if len(batch_X) == 0:
            # Return empty batch if no valid videos
            return np.empty((0, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)), np.empty((0, NUM_CLASSES))
        
        return np.array(batch_X, dtype=np.float32), to_categorical(batch_y, num_classes=NUM_CLASSES)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================
# 5. GET VIDEO PATHS (NOT LOADING INTO MEMORY)
# ============================================================

print("\nScanning Training Data...")
train_video_paths, train_labels = get_video_paths(TRAIN_DIR)

print("\nScanning Testing Data...")
test_video_paths, test_labels = get_video_paths(TEST_DIR)

print(f"\nDataset Info:")
print(f"Training videos: {len(train_video_paths)}")
print(f"Testing videos: {len(test_video_paths)}")

# Create data generators
train_generator = VideoDataGenerator(train_video_paths, train_labels, BATCH_SIZE, shuffle=True)
test_generator = VideoDataGenerator(test_video_paths, test_labels, BATCH_SIZE, shuffle=False)

# ============================================================
# 6. BUILD CNN + LSTM MODEL
# ============================================================

print("\nBuilding Model...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze most layers (but unfreeze last few for better feature learning)
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

model = Sequential()

model.add(TimeDistributed(
    base_model,
    input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)
))

model.add(TimeDistributed(GlobalAveragePooling2D()))
model.add(LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.001)))
model.add(Dropout(0.6))
model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.001)))
model.add(Dropout(0.4))
model.add(Dense(NUM_CLASSES, activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 7. TRAINING CALLBACKS (PREVENT OVERFITTING)
# ============================================================

# Early Stopping: Stop training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

# Model Checkpoint: Save the best model
checkpoint = ModelCheckpoint(
    'best_hawk_eye_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Reduce Learning Rate: Lower LR when validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nTraining Model with Early Stopping...")
print(f"- Early stopping patience: {PATIENCE} epochs")
print("- Best model will be saved automatically")
print("- Learning rate will adapt if validation loss plateaus\n")

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# ============================================================
# 9. SAVE FINAL MODEL
# ============================================================

MODEL_PATH = "hawk_eye_threat_detector_final.h5"
model.save(MODEL_PATH)

print(f"\nFinal model saved at: {MODEL_PATH}")
print(f"Best model saved at: best_hawk_eye_model.h5")

# ============================================================
# 10. EVALUATE MODEL
# ============================================================

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

loss, accuracy = model.evaluate(test_generator)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Training history summary
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Total epochs trained: {len(history.history['loss'])}")
print(f"Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
print(f"Best validation loss: {min(history.history['val_loss']):.4f}")

# Save training history for later analysis
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("\nTraining history saved to 'training_history.pkl'")

# ============================================================
# 11. PREDICTION FUNCTION
# ============================================================

def predict_video(video_path):
    frames = extract_frames(video_path)

    if frames is None:
        return "Error: Could not process video"

    frames = np.expand_dims(frames, axis=0)

    prediction = model.predict(frames)[0]
    class_index = np.argmax(prediction)

    return CLASSES[class_index]

# ============================================================
# 12. TEST PREDICTION (OPTIONAL)
# ============================================================

# Example usage:
# video_test_path = r"test_video.mp4"
# result = predict_video(video_test_path)
# print("Prediction:", result)

# ============================================================
# END OF FILE — READY TO TRAIN 🚀
# ============================================================