# Deepfake Detection System — CPU Starter

This is a runnable starter application, not yet a trained deepfake classifier.

## What works now
- Flask web dashboard
- Video upload and metadata inspection
- OpenCV frame sampling
- Face detection using OpenCV Haar cascade
- Browser webcam preview and sampled-frame face detection
- SQLite analysis history
- CSV dataset inspection

## Important limitation
The uploaded CSV has columns `id`, `deepfake`, `image`, and `video`, but it does not contain an explicit `real/fake` label column. The `deepfake` path appears to reference manipulated clips, but the meaning of `video` must be verified from the dataset documentation. The application deliberately does not guess labels or show fake/real predictions before a model is trained and validated.

## Python
Python 3.14 is supported by current PyTorch Windows installation guidance. Install matching current wheels using pip. If a third-party package fails to install, use Python 3.13 as a fallback rather than changing the project code blindly.

## Setup on Windows (PowerShell)
Open this folder in VS Code, then Terminal > New Terminal:

```powershell
py -3.14 --version
py -3.14 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install CPU PyTorch if pip selected a different build or you want to explicitly choose CPU:

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If PowerShell blocks activation, run this for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Put the CSV and media in the dataset folder
Expected layout:

```text
dataset/
  DeepFake Videos Dataset.csv
  deepfake/   # files referenced by the deepfake column
  video/      # files referenced by the video column
  image/      # files referenced by the image column
```

Do not rename media files unless you also update the CSV paths.

## Inspect dataset
From the project root:

```powershell
python training/inspect_dataset.py
```

Creates `results/dataset_report.csv`.

## Run the web application

```powershell
python backend/app.py
```

Open http://127.0.0.1:5000 in your browser.

Use Stop (Ctrl+C) in the terminal to stop the server.

## Next steps before real/fake training
1. Read the dataset documentation and verify what the `video` and `deepfake` columns mean.
2. Confirm that each source video has a trustworthy REAL or FAKE label.
3. Create a video-level split (not a random frame-level split) to prevent leakage.
4. Extract frames from each split into `dataset/frames/{real,fake}`.
5. Add face cropping and a complete EfficientNet training/evaluation pipeline.
6. Integrate the trained model architecture and preprocessing into `backend/detector.py`.

A face detector only locates faces. It does not decide whether a face/video is a deepfake.
