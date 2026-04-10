import numpy as np
import math
import cv2
from typing import List, Tuple, Dict
from .config import get_settings
from .models import predict_batch, get_model_input_size, get_model_num_classes

settings = get_settings()

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def entropy(probs: np.ndarray) -> float:
    return -float(np.sum(probs * np.log(probs + 1e-9)))

def sliding_window(frame: np.ndarray) -> Tuple[List[Tuple[int,int,int,int]], np.ndarray]:
    h, w, _ = frame.shape
    win = settings.window_size
    stride = settings.stride
    windows = []
    patches = []
    input_h, input_w = get_model_input_size()
    for y in range(0, h - win + 1, stride):
        for x in range(0, w - win + 1, stride):
            crop = frame[y:y+win, x:x+win]
            # Resize crop to model input size if different
            if crop.shape[0] != input_h or crop.shape[1] != input_w:
                crop = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            # Convert BGR to RGB (OpenCV uses BGR but MobileNetV2 expects RGB)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            patches.append(crop)
            windows.append((x, y, x+win, y+win))
    if not patches:
        return [], np.zeros((0, get_model_num_classes()))
    batch = np.array([p.astype(np.float32)/255.0 for p in patches])
    preds = predict_batch(batch)  # assume output shape (N, num_classes)
    return windows, preds

def filter_detections(windows: List[Tuple[int,int,int,int]], preds: np.ndarray) -> List[Dict]:
    detections = []
    for bbox, probs in zip(windows, preds):
        # If probs don't sum close to 1, apply softmax; otherwise trust model output.
        s = float(np.sum(probs))
        if not (0.999 <= s <= 1.001):
            probs = softmax(probs)
        conf = float(np.max(probs))
        cls_idx = int(np.argmax(probs))
        cls_name = settings.classes[cls_idx]
        ent = entropy(probs)
        
        # Only report Violence and Weaponized as detections (skip Normal)
        # Trust the model's prediction without threshold filtering
        if cls_name in ["Violence", "Weaponized"]:
            detections.append({
                'class': cls_name,
                'confidence': conf,
                'entropy': ent,
                'bbox': bbox,
                'probs': probs
            })
    return detections

def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter_area / (a_area + b_area - inter_area + 1e-9)

def nms(detections: List[Dict]) -> List[Dict]:
    # group by class
    final = []
    for cls in settings.classes:
        cls_dets = [d for d in detections if d['class'] == cls]
        cls_dets.sort(key=lambda d: d['confidence'], reverse=True)
        kept = []
        for det in cls_dets:
            if all(iou(det['bbox'], k['bbox']) < settings.nms_iou_threshold for k in kept):
                kept.append(det)
        final.extend(kept)
    return final

def detect_frame(frame: np.ndarray) -> Dict:
    windows, preds = sliding_window(frame)
    detections = filter_detections(windows, preds)
    detections = nms(detections)
    threat_counts = {}
    for d in detections:
        threat_counts[d['class']] = threat_counts.get(d['class'], 0) + 1
    return {
        'detections': [
            {
                'class': d['class'],
                'confidence': d['confidence'],
                'bbox': list(d['bbox'])
            } for d in detections
        ],
        'summary': {
            'windows_scanned': len(windows),
            'threat_counts': threat_counts
        }
    }
