"""
Detection logging utility for saving webcam detection sessions
"""
import json
import os
from datetime import datetime
from typing import Dict, List
from pathlib import Path

class DetectionLogger:
    def __init__(self, logs_dir: str):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = None
        self.session_file = None
        
    def start_session(self, session_id: str = None):
        """Start a new detection session"""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.current_session = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'detections': [],
            'stats': {
                'total_frames': 0,
                'frames_with_threats': 0,
                'total_threats': 0,
                'threat_counts': {}
            }
        }
        
        self.session_file = self.logs_dir / f"session_{session_id}.json"
        return session_id
    
    def log_detection(self, frame_number: int, detections: List[Dict], summary: Dict):
        """Log a single frame's detection results"""
        if self.current_session is None:
            return
        
        self.current_session['stats']['total_frames'] += 1
        
        if detections:
            self.current_session['stats']['frames_with_threats'] += 1
            self.current_session['stats']['total_threats'] += len(detections)
            
            # Update threat counts
            for det in detections:
                threat_class = det['class']
                self.current_session['stats']['threat_counts'][threat_class] = \
                    self.current_session['stats']['threat_counts'].get(threat_class, 0) + 1
            
            # Log detection event
            self.current_session['detections'].append({
                'timestamp': datetime.now().isoformat(),
                'frame_number': frame_number,
                'detections': detections,
                'summary': summary
            })
    
    def end_session(self):
        """End current session and save to file"""
        if self.current_session is None:
            return None
        
        self.current_session['end_time'] = datetime.now().isoformat()
        
        # Calculate session duration
        start = datetime.fromisoformat(self.current_session['start_time'])
        end = datetime.fromisoformat(self.current_session['end_time'])
        duration = (end - start).total_seconds()
        self.current_session['duration_seconds'] = duration
        
        # Save to file
        with open(self.session_file, 'w') as f:
            json.dump(self.current_session, f, indent=2)
        
        session_id = self.current_session['session_id']
        self.current_session = None
        self.session_file = None
        
        return session_id
    
    def get_session_summary(self):
        """Get current session summary"""
        if self.current_session is None:
            return None
        return {
            'session_id': self.current_session['session_id'],
            'start_time': self.current_session['start_time'],
            'stats': self.current_session['stats']
        }
    
    def list_sessions(self, limit: int = 10):
        """List recent detection sessions"""
        session_files = sorted(self.logs_dir.glob("session_*.json"), reverse=True)
        sessions = []
        
        for session_file in session_files[:limit]:
            try:
                with open(session_file, 'r') as f:
                    session = json.load(f)
                    sessions.append({
                        'session_id': session['session_id'],
                        'start_time': session['start_time'],
                        'end_time': session.get('end_time'),
                        'duration_seconds': session.get('duration_seconds'),
                        'stats': session['stats']
                    })
            except Exception:
                continue
        
        return sessions
    
    def get_session(self, session_id: str):
        """Get full session data by ID"""
        session_file = self.logs_dir / f"session_{session_id}.json"
        if not session_file.exists():
            return None
        
        with open(session_file, 'r') as f:
            return json.load(f)
    
    def delete_session(self, session_id: str):
        """Delete a session log file"""
        session_file = self.logs_dir / f"session_{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            return True
        return False
