import os
from functools import lru_cache

class Settings:
    def __init__(self) -> None:
        default_model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'best_hawk_eye_model.h5')
        )
        env_model_path = os.getenv('MODEL_PATH')
        self.model_path: str = env_model_path if env_model_path else default_model_path

        env_classes = os.getenv('MODEL_CLASSES')
        if env_classes:
            self.classes = [c.strip() for c in env_classes.split(',') if c.strip()]
        else:
            # Class names matching training data (SCVD_converted dataset)
            self.classes = ['Normal', 'Violence', 'Weaponized']

        self.window_size: int = int(os.getenv('WINDOW_SIZE', '192'))
        self.stride: int = int(os.getenv('WINDOW_STRIDE', '150'))  # Increased for faster processing
        self.confidence_threshold: float = float(os.getenv('CONF_THRESHOLD', '0.50'))  # Lower threshold for 3-class model
        self.entropy_threshold: float = float(os.getenv('ENTROPY_THRESHOLD', '1.10'))
        self.nms_iou_threshold: float = float(os.getenv('NMS_IOU_THRESHOLD', '0.3')) 
        self.frame_skip: int = int(os.getenv('FRAME_SKIP', '2'))  # Process every Nth frame
        self.processed_dir: str = os.path.abspath(
            os.getenv('PROCESSED_DIR', os.path.join(os.path.dirname(__file__), '..', 'processed'))
        )

@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    os.makedirs(s.processed_dir, exist_ok=True)
    return s
