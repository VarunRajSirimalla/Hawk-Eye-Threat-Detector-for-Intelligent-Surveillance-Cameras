import os
import tensorflow as tf
# Use tf_keras for Keras 2 model compatibility
try:
    import tf_keras as keras
    print("Using tf_keras (Keras 2 compatibility mode)")
except ImportError:
    keras = tf.keras
    print("Using tf.keras (Keras 3)")
import numpy as np
from .config import get_settings

settings = get_settings()

# GPU/CPU selection and memory growth
physical_gpus = tf.config.list_physical_devices('GPU')
for gpu in physical_gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

MODEL = None
MODEL_ERROR = None
DISABLE_MODEL = os.getenv('DISABLE_MODEL', '0') in ('1', 'true', 'True')

def rebuild_model_architecture():
    """Rebuild the model architecture to match training (fallback for compatibility issues)"""
    try:
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import (
            TimeDistributed, GlobalAveragePooling2D, LSTM, Dense, Dropout
        )
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.optimizers import Adam
        
        IMG_SIZE = 224
        SEQUENCE_LENGTH = 30
        NUM_CLASSES = len(settings.classes)
        
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
    except Exception as e:
        print(f"Failed to rebuild model architecture: {e}")
        return None

def _load_model():
    global MODEL, MODEL_ERROR
    if DISABLE_MODEL:
        return None
    if MODEL is not None:
        return MODEL
    
    print(f"Loading model from: {settings.model_path}")
    
    # Try loading with keras (tf_keras for Keras 2 models)
    try:
        MODEL = keras.models.load_model(settings.model_path, compile=False)
        from tensorflow.keras.optimizers import Adam
        MODEL.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )
        print(f"✓ Model loaded successfully!")
        print(f"Model input shape: {MODEL.input_shape}")
        print(f"Model output shape: {MODEL.output_shape}")
        return MODEL
    except Exception as e:
        print(f"⚠ Compatibility issue detected: {type(e).__name__}")
        print(f"Error: {str(e)[:200]}")
        print("🔧 Rebuilding model architecture and loading weights...")
        
        try:
            # Rebuild model and load weights
            MODEL = rebuild_model_architecture()
            if MODEL is None:
                raise Exception("Failed to rebuild architecture")
            MODEL.load_weights(settings.model_path)
            print("✓ Model loaded successfully with weights!")
            print(f"Model input shape: {MODEL.input_shape}")
            print(f"Model output shape: {MODEL.output_shape}")
            MODEL_ERROR = None
            return MODEL
        except Exception as e2:
            MODEL_ERROR = e2
            print(f"❌ Failed to load weights: {str(e2)}")
            print("Backend will continue with zero predictions.")
            MODEL = None
            return None

def predict_batch(batch: np.ndarray) -> np.ndarray:
    if DISABLE_MODEL:
        return np.zeros((batch.shape[0], len(settings.classes)), dtype=np.float32)
    m = _load_model()
    if m is None:
        return np.zeros((batch.shape[0], len(settings.classes)), dtype=np.float32)
    try:
        return m.predict(batch, verbose=0)
    except Exception as e:
        # Capture runtime prediction errors and degrade gracefully
        global MODEL_ERROR
        MODEL_ERROR = e
        print(f"Prediction error: {e}. Returning zeros.")
        return np.zeros((batch.shape[0], len(settings.classes)), dtype=np.float32)

def model_status() -> dict:
    return {
        "disabled": DISABLE_MODEL,
        "loaded": MODEL is not None and MODEL_ERROR is None and not DISABLE_MODEL,
        "error": None if MODEL_ERROR is None else str(MODEL_ERROR),
        "model_path": settings.model_path,
        "classes": settings.classes,
    }

def get_model_input_size() -> tuple:
    m = _load_model()
    if m is None or not m.input_shape or len(m.input_shape) != 4:
        return (settings.window_size, settings.window_size)
    return (m.input_shape[1], m.input_shape[2])

def get_model_num_classes() -> int:
    m = _load_model()
    if m is None or not m.output_shape:
        return len(settings.classes)
    return int(m.output_shape[-1])
