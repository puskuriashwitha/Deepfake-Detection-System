
# Deepfake Detection System

A real-time application for deepfake detection in live video communication systems.

## Features

- Video upload and deepfake analysis
- Live webcam face detection and frame analysis
- Two-person video communication
- Room creation and joining
- Analysis history
- CNN-based experimental deepfake classification

## Technologies Used

- Python
- Flask
- OpenCV
- PyTorch
- HTML, CSS and JavaScript
- WebRTC
- SQLite

## Project Structure

```text
DEEPFAKE_DETECTION_SYSTEM/
├── backend/
├── frontend/
├── models/
├── training/
├── results/
├── requirements.txt
└── README.md
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python backend/app.py
```

Open:

http://127.0.0.1:5000/

## Note

The deepfake classification model is experimental and requires further validation.
