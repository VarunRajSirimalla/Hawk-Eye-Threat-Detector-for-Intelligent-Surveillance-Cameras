// DOM Elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const logEl = document.getElementById('log');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const framesProcessedEl = document.getElementById('framesProcessed');
const threatsDetectedEl = document.getElementById('threatsDetected');
const windowsScannedEl = document.getElementById('windowsScanned');
const detectionsListEl = document.getElementById('detectionsList');

const ctx = canvas.getContext('2d');
let ws = null;
let streaming = false;
let frameCount = 0;
let totalThreats = 0;
let totalWindows = 0;
let recentDetections = [];
let currentSessionId = null;

// Colors for different threat types
const THREAT_COLORS = {
    'Normal': '#10b981',      // Green for normal/safe
    'Violence': '#ef4444',    // Red for violence
    'Weaponized': '#dc2626'   // Dark red for weaponized threats
};

// Initialize webcam
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        video.srcObject = stream;
        logMessage('Webcam initialized successfully', 'info');
    } catch (err) {
        logMessage(`Failed to access webcam: ${err.message}`, 'error');
        alert('Unable to access webcam. Please check permissions.');
    }
}

// Draw detections with bounding boxes
function drawDetections(dets) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!dets || dets.length === 0) return;
    
    // Scale factor to match video display size to canvas size
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;
    
    dets.forEach(d => {
        const [x1, y1, x2, y2] = d.bbox;
        const color = THREAT_COLORS[d.class] || '#ef4444';
        
        // Scale coordinates
        const sx1 = x1 * scaleX;
        const sy1 = y1 * scaleY;
        const sw = (x2 - x1) * scaleX;
        const sh = (y2 - y1) * scaleY;
        
        // Draw rectangle
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(sx1, sy1, sw, sh);
        
        // Draw label background
        const label = `${d.class} ${(d.confidence * 100).toFixed(1)}%`;
        ctx.font = 'bold 14px sans-serif';
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(sx1, sy1 - 24, textWidth + 12, 24);
        
        // Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(label, sx1 + 6, sy1 - 6);
    });
}

// Capture and send frame to backend
function captureFrame() {
    if (!streaming) return;
    
    const off = document.createElement('canvas');
    off.width = video.videoWidth;
    off.height = video.videoHeight;
    off.getContext('2d').drawImage(video, 0, 0);
    const b64 = off.toDataURL('image/jpeg', 0.75);
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(b64);
        frameCount++;
        framesProcessedEl.textContent = frameCount;
    }
    
    requestAnimationFrame(captureFrame);
}

// Log message with timestamp
function logMessage(msg, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const prefix = type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️';
    const logLine = `[${timestamp}] ${prefix} ${msg}\n`;
    logEl.textContent = logLine + logEl.textContent;
}

// Update detection stats
function updateStats(data) {
    if (data.detections && data.detections.length > 0) {
        totalThreats += data.detections.length;
        threatsDetectedEl.textContent = totalThreats;
        
        // Add to recent detections
        data.detections.forEach(d => {
            const detection = {
                timestamp: new Date().toLocaleTimeString(),
                class: d.class,
                confidence: d.confidence
            };
            recentDetections.unshift(detection);
        });
        
        // Keep only last 10
        recentDetections = recentDetections.slice(0, 10);
        updateDetectionsList();
    }
    
    if (data.summary && data.summary.windows_scanned) {
        totalWindows = data.summary.windows_scanned;
        windowsScannedEl.textContent = totalWindows;
    }
}

// Update recent detections list
function updateDetectionsList() {
    if (recentDetections.length === 0) {
        detectionsListEl.innerHTML = '<p class=\"text-sm text-gray-500 italic\">No detections yet...</p>';
        return;
    }
    
    detectionsListEl.innerHTML = recentDetections.map(d => {
        const color = THREAT_COLORS[d.class] || '#ef4444';
        return `
            <div class="flex items-center justify-between p-2 bg-gray-50 rounded text-xs">
                <div class="flex items-center gap-2">
                    <div style="width:8px;height:8px;background:${color};border-radius:50%"></div>
                    <span class="font-semibold">${d.class}</span>
                </div>
                <div class="text-right">
                    <div class="font-mono">${(d.confidence * 100).toFixed(1)}%</div>
                    <div class="text-gray-500">${d.timestamp}</div>
                </div>
            </div>
        `;
    }).join('');
}

// Update UI status
function updateStatus(status, message) {
    statusDot.className = `status-indicator status-${status}`;
    statusText.textContent = message;
}

// Start detection
startBtn.addEventListener('click', () => {
    if (ws) ws.close();
    
    const API_BASE = (window.API_BASE || 'http://localhost:8001');
    const wsUrl = API_BASE.replace('http', 'ws') + '/ws/webcam';
    
    updateStatus('processing', 'Connecting...');
    logMessage(`Connecting to ${wsUrl}`, 'info');
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        streaming = true;
        updateStatus('connected', 'Connected');
        logMessage('WebSocket connected. Starting detection...', 'info');
        startBtn.disabled = true;
        stopBtn.disabled = false;
        captureFrame();
    };
    
    ws.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            
            // Handle session start
            if (data.type === 'session_start') {
                currentSessionId = data.session_id;
                logMessage(`Session started: ${currentSessionId}`, 'info');
                return;
            }
            
            if (data.error) {
                logMessage(`Error: ${data.error}`, 'error');
                return;
            }
            
            if (data.detections) {
                drawDetections(data.detections);
                updateStats(data);
                
                if (data.detections.length > 0) {
                    logMessage(`Detected ${data.detections.length} threat(s): ${data.detections.map(d => d.class).join(', ')}`, 'warning');
                }
            }
        } catch (err) {
            logMessage(`Parse error: ${err.message}`, 'error');
        }
    };
    
    ws.onerror = (err) => {
        logMessage('WebSocket error occurred', 'error');
        updateStatus('disconnected', 'Error');
    };
    
    ws.onclose = () => {
        streaming = false;
        updateStatus('disconnected', 'Disconnected');
        if (currentSessionId) {
            logMessage(`Session ended: ${currentSessionId}. View logs at: /logs/session/${currentSessionId}`, 'info');
        } else {
            logMessage('WebSocket closed', 'info');
        }
        startBtn.disabled = false;
        stopBtn.disabled = true;
    };
});

// Stop detection
stopBtn.addEventListener('click', () => {
    streaming = false;
    if (ws) {
        ws.close();
        logMessage('Detection stopped by user', 'info');
    }
});

// Initialize
startWebcam();
