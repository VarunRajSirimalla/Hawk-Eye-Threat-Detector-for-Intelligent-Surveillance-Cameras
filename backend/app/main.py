import os
import io
import cv2
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Dict, List
from .config import get_settings
from .detection import detect_frame
from .models import model_status
from .schemas import UploadVideoResponse, FrameDetectRequest, FrameDetectResponse, RTSPRequest, RTSPAck
from .utils.video import read_video_frames, encode_frame_jpeg, open_rtsp_stream
from .logger import DetectionLogger

settings = get_settings()
app = FastAPI(title="CCTV Threat Detection System", version="1.0.0")

# Initialize detection logger
logs_dir = os.path.join(settings.processed_dir, 'detection_logs')
logger = DetectionLogger(logs_dir)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/health/model")
def health_model():
    return model_status()

DEBUG_DETECTIONS = os.getenv('DEBUG_DETECTIONS', '0') in ('1','true','True')

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    tmp_path = os.path.join(settings.processed_dir, file.filename)
    with open(tmp_path, 'wb') as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        return JSONResponse(status_code=400, content={"error": "Cannot open video"})

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Extract frames as sequence (matching test_model.py logic)
    # LSTM model expects sequences of 30 frames
    from .models import predict_batch
    SEQUENCE_LENGTH = 30
    IMG_SIZE = 224
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // SEQUENCE_LENGTH, 1)
    
    frames_sequence = []
    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Preprocess exactly like test_model.py
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        frames_sequence.append(img)
    
    cap.release()
    
    if DEBUG_DETECTIONS:
        print(f"\n=== Video Processing ===")
        print(f"Total frames in video: {total_frames}")
        print(f"Extracted sequence length: {len(frames_sequence)}")
        print(f"Sequence shape per frame: {frames_sequence[0].shape if frames_sequence else 'N/A'}")

    
    # Make prediction on sequence (matching test_model.py logic)
    if len(frames_sequence) == SEQUENCE_LENGTH:
        frames_array = np.array(frames_sequence, dtype=np.float32)
        frames_batch = np.expand_dims(frames_array, axis=0)  # Shape: (1, 30, 224, 224, 3)
        
        if DEBUG_DETECTIONS:
            print(f"Input batch shape: {frames_batch.shape}")
        
        predictions = predict_batch(frames_batch)[0]  # Get predictions for single video
        
        class_id = int(np.argmax(predictions))
        confidence = float(predictions[class_id])
        predicted_class = settings.classes[class_id]
        all_probs = {settings.classes[i]: float(predictions[i]) for i in range(len(predictions))}
        
        if DEBUG_DETECTIONS:
            print(f"\n=== Prediction Results ===")
            print(f"Class probabilities:")
            for cls, prob in all_probs.items():
                marker = " ← PREDICTED" if cls == predicted_class else ""
                print(f"  {cls:12} : {prob:.4f} ({prob*100:.2f}%){marker}")
            print(f"Final prediction: {predicted_class}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        final_label = predicted_class
    else:
        if DEBUG_DETECTIONS:
            print(f"⚠ Warning: Could not extract full sequence (got {len(frames_sequence)}/{SEQUENCE_LENGTH})")
        final_label = "NO PREDICTION"
        confidence = 0.0
        all_probs = {}
        predicted_class = "Unknown"
    
    # Create annotated video
    out_path_annotated = os.path.join(settings.processed_dir, f"final_{file.filename}.mp4")
    cap2 = cv2.VideoCapture(tmp_path)
    writer2 = cv2.VideoWriter(out_path_annotated, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        # Add final prediction overlay
        color = (0, 255, 0) if final_label == "Normal" else (0, 0, 255)
        cv2.putText(frame, f"PREDICTION: {final_label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"CONFIDENCE: {confidence:.2%}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        writer2.write(frame)
    
    cap2.release()
    writer2.release()
    
    # Determine if it's a threat: Violence and Weaponized are threats, Normal is safe
    is_threat = final_label in ["Violence", "Weaponized"]
    threat_type = final_label if is_threat else None
    
    response = UploadVideoResponse(
        detections=[{
            'class_': final_label,
            'confidence': confidence,
            'bbox': [0, 0, 0, 0]  # No bounding box for full video classification
        }],
        summary={
            'frames_total': total_frames,
            'frames_processed': len(frames_sequence),
            'final_prediction': final_label,
            'confidence': confidence,
            'all_probabilities': all_probs if len(frames_sequence) == SEQUENCE_LENGTH else {},
            'is_threat': is_threat,
            'threat_counts': {threat_type: 1} if is_threat else {}
        },
        processed_video=f"/processed/final_{file.filename}.mp4"
    )
    return response

@app.get('/processed/{filename}')
async def get_processed(filename: str):
    path = os.path.join(settings.processed_dir, filename)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(path, media_type='video/mp4')

@app.post('/detect-frame')
async def detect_frame_endpoint(payload: FrameDetectRequest):
    try:
        raw = base64.b64decode(payload.image_base64.split(',')[-1])
        data = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        result = detect_frame(img)
        detections = [
            {'class_': d['class'], 'confidence': d['confidence'], 'bbox': d['bbox']} for d in result['detections']
        ]
        return FrameDetectResponse(detections=detections, summary=result['summary']['threat_counts'])
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# WebSocket for webcam: client sends base64 frames, server returns JSON detections
@app.websocket('/ws/webcam')
async def ws_webcam(ws: WebSocket):
    await ws.accept()
    session_id = logger.start_session()
    frame_count = 0
    
    try:
        # Send session ID to client
        await ws.send_json({'type': 'session_start', 'session_id': session_id})
        
        while True:
            msg = await ws.receive_text()
            if msg == 'ping':
                await ws.send_text('pong')
                continue
            # Expect base64 frame
            try:
                frame_count += 1
                raw = base64.b64decode(msg.split(',')[-1])
                data = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                result = detect_frame(img)
                
                # Log detection
                logger.log_detection(frame_count, result.get('detections', []), result.get('summary', {}))
                
                # Add session info to response
                result['session_id'] = session_id
                result['frame_number'] = frame_count
                
                await ws.send_json(result)
            except Exception as e:
                await ws.send_json({'error': str(e)})
    except WebSocketDisconnect:
        # End session when client disconnects
        logger.end_session()
        pass

# WebSocket for RTSP: first message is URL, then stream frames back with detections
@app.websocket('/ws/rtsp')
async def ws_rtsp(ws: WebSocket):
    await ws.accept()
    try:
        url = await ws.receive_text()
        for frame in open_rtsp_stream(url):
            result = detect_frame(frame)
            # annotate frame
            for d in result['detections']:
                x1,y1,x2,y2 = d['bbox']
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, d['class'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            b64 = encode_frame_jpeg(frame)
            await ws.send_json({'frame': b64, 'detections': result['detections'], 'summary': result['summary']})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_json({'error': str(e)})


# Detection logs management endpoints
@app.get('/logs/sessions')
async def list_detection_sessions(limit: int = 20):
    """List recent detection sessions"""
    sessions = logger.list_sessions(limit=limit)
    return {'sessions': sessions, 'total': len(sessions)}

@app.get('/logs/session/{session_id}')
async def get_detection_session(session_id: str):
    """Get detailed information for a specific session"""
    session = logger.get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={'error': 'Session not found'})
    return session

@app.delete('/logs/session/{session_id}')
async def delete_detection_session(session_id: str):
    """Delete a detection session log"""
    success = logger.delete_session(session_id)
    if not success:
        return JSONResponse(status_code=404, content={'error': 'Session not found'})
    return {'message': 'Session deleted successfully', 'session_id': session_id}

@app.get('/logs/current')
async def get_current_session():
    """Get current active session summary"""
    summary = logger.get_session_summary()
    if summary is None:
        return {'active': False}
    return {'active': True, **summary}
