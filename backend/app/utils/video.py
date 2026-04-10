import cv2
import base64
import numpy as np
from typing import Generator

def read_video_frames(path: str) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def encode_frame_jpeg(frame: np.ndarray, quality: int = 70) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buf = cv2.imencode('.jpg', frame, encode_param)
    if not ok:
        return ''
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def open_rtsp_stream(url: str) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()
