from typing import List, Dict, Optional
from pydantic import BaseModel

class Detection(BaseModel):
    class_: str
    confidence: float
    bbox: List[int]

class DetectionSummary(BaseModel):
    frames_processed: int
    threat_counts: Dict[str, int]

class UploadVideoResponse(BaseModel):
    detections: List[Detection]
    summary: DetectionSummary
    processed_video: str

class FrameDetectRequest(BaseModel):
    image_base64: str

class FrameDetectResponse(BaseModel):
    detections: List[Detection]
    summary: Dict[str, int]

class RTSPRequest(BaseModel):
    url: str

class RTSPAck(BaseModel):
    status: str
    message: str
