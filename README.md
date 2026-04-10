# Hawk-Eye: AI-Powered CCTV Threat Detection

Hawk-Eye is a deep learning project for CCTV threat detection with:
- A CNN + LSTM model for classifying video clips into `Normal`, `Violence`, and `Weaponized`
- Testing scripts for single-video and full-dataset evaluation
- A FastAPI + HTML/JS web application for upload, webcam, and RTSP-based detection

## Project Highlights

- Multi-class threat classification on the SCVD-converted dataset
- Training pipeline based on MobileNetV2 + LSTM
- Evaluation tools with confusion matrix and classification report
- Web interface with API backend and static frontend
- Ready-to-run PowerShell launcher for demo/presentation flow

## Repository Structure

```text
.
├── hawk_eye_threat_detection_starter.py    # Training script
├── quick_test.py                           # Fast single-video prediction
├── test_model.py                           # Full evaluation + metrics
├── best_hawk_eye_model.h5                  # Best checkpoint model
├── hawk_eye_threat_detector_final.h5       # Final epoch model
├── final_violence_detection_model.h5       # Alternate trained model file
├── SCVD_converted/
│   ├── Train/
│   │   ├── Normal/
│   │   ├── Violence/
│   │   └── Weaponized/
│   └── Test/
│       ├── Normal/
│       ├── Violence/
│       └── Weaponized/
└── webapp/
    ├── backend/                            # FastAPI app
    ├── frontend/                           # Static HTML/CSS/JS pages
    ├── start_app.ps1                       # Start backend + frontend together
    ├── stop_app.ps1                        # Stop running app processes
    └── QUICK_START.md                      # Demo-first startup guide
```

## Requirements

- Python 3.10+ (recommended)
- Windows PowerShell (for bundled `.ps1` scripts)
- Optional: Docker Desktop (for compose workflow)

Main Python libraries used:
- TensorFlow
- OpenCV
- NumPy
- FastAPI
- Uvicorn

## Setup (Root Project)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install tensorflow opencv-python numpy scikit-learn matplotlib seaborn
```

## Train the Model

```powershell
python hawk_eye_threat_detection_starter.py
```

Expected outputs include:
- `best_hawk_eye_model.h5`
- `hawk_eye_threat_detector_final.h5`

## Test and Evaluate

### Quick single-video test

```powershell
python quick_test.py "SCVD_converted/Test/Violence/V_001.mp4"
```

### Detailed single-video test

```powershell
python test_model.py --video "SCVD_converted/Test/Violence/V_001.mp4"
```

### Full test-set evaluation

```powershell
python test_model.py --full-test
```

Useful options:

```powershell
python test_model.py --model "hawk_eye_threat_detector_final.h5" --full-test
python test_model.py --help
```

## Run the Web Application

The web app uses:
- Backend: FastAPI on port `8001`
- Frontend: static server on port `5173`

### Recommended (one-command start)

```powershell
cd webapp
.\start_app.ps1
```

Then open:
- `http://localhost:5173/upload.html`
- `http://localhost:5173/webcam.html`
- API docs: `http://localhost:8001/docs`

Stop with `Ctrl+C` in that terminal, or run:

```powershell
.\stop_app.ps1
```

### Manual start (optional)

Backend:

```powershell
cd webapp\backend
pip install -r requirements.txt
python run_backend.py
```

Frontend (new terminal):

```powershell
cd webapp
python run_frontend.py
```

## WebApp Capabilities

- Upload video for processed detection output
- Real-time webcam detection via WebSocket
- RTSP stream ingestion with detection overlays
- Health endpoint for readiness checks

## Model Selection Notes

- The backend defaults to `best_hawk_eye_model.h5`
- You can override model path using environment variable `MODEL_PATH`
- `quick_test.py` prefers `best_hawk_eye_model.h5` and falls back to `hawk_eye_threat_detector_final.h5`
- `test_model.py` defaults to `final_violence_detection_model.h5` unless `--model` is provided

## Additional Documentation

See these documents for deeper details:
- `TESTING_GUIDE.md`
- `CONFERENCE_PAPER_DESCRIPTION.md`
- `webapp/README.md`
- `webapp/QUICK_START.md`
- `webapp/WEBCAM_DETECTION_GUIDE.md`
- `webapp/DETECTION_LOGGING_GUIDE.md`

## Troubleshooting

- If a port is busy, run `webapp\stop_app.ps1` and start again.
- If model loading fails, verify the `.h5` file exists in project root.
- If predictions are unstable, test with higher-quality videos and sufficient frame count.
- If frontend cannot reach backend, confirm backend health at `http://localhost:8001/health`.
