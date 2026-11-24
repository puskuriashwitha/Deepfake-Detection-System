
from pathlib import Path
import sys
import uuid
import threading
import secrets
import time
import cv2
import numpy as np

from flask import (
    Flask,
    request,
    jsonify,
    render_template
)

from flask_cors import CORS


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from backend.face_detector import FaceDetector
from backend.video_processor import inspect_video
from backend.detector import DeepfakeDetector
from backend.database import (
    init_db,
    add_analysis,
    get_history
)


UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm"
}


app = Flask(
    __name__,
    template_folder="../frontend",
    static_folder="../frontend",
    static_url_path="/static"
)

app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024

CORS(app)


# Initialize components
face_detector = FaceDetector()
deepfake_detector = DeepfakeDetector()

init_db()


def classify_video_frames(sampled_frames):
    """
    Predict every sampled complete frame and
    calculate the average fake probability.
    """

    predictions = []

    for frame_index, frame in sampled_frames:

        prediction = deepfake_detector.predict(frame)

        if prediction["status"] == "prediction_available_experimental":
            predictions.append(prediction)

    if not predictions:

        return {
            "classification": "NOT AVAILABLE",
            "confidence": None,
            "real_probability": None,
            "fake_probability": None,
            "status": "model_unavailable",
            "predicted_frames": 0,
            "message": (
                "No valid frame predictions were available."
            )
        }

    # Average probabilities across sampled frames
    average_fake = float(
        np.mean([
            item["fake_probability"]
            for item in predictions
        ])
    )

    average_real = 1.0 - average_fake

    if average_fake >= 0.5:
        classification = "FAKE"
        confidence = average_fake
    else:
        classification = "REAL"
        confidence = average_real

    return {
        "classification": classification,
        "confidence": round(confidence, 4),
        "real_probability": round(average_real, 4),
        "fake_probability": round(average_fake, 4),
        "status": "prediction_available_experimental",
        "predicted_frames": len(predictions),
        "message": (
            "Experimental prediction based on average "
            "CNN probabilities across sampled video frames. "
            "Not a reliable authenticity verdict."
        )
    }


@app.get("/")
def home():
    return render_template("index.html")

@app.get("/call")
def call_page():
    return render_template("call.html")

@app.get("/api/health")
def health():

    return jsonify({
        "app": "Deepfake Detection System",
        "status": "running",
        "device": "CPU",
        "classifier_ready": deepfake_detector.model_ready,
        "model_error": deepfake_detector.model_error,
        "note": (
            "Experimental CNN classifier. "
            "Predictions require further validation."
        )
    })


@app.post("/api/analyze-video")
def analyze_video():

    uploaded = request.files.get("video")

    if not uploaded or not uploaded.filename:
        return jsonify({
            "error": "Choose a video file first."
        }), 400

    suffix = Path(uploaded.filename).suffix.lower()

    if suffix not in ALLOWED:
        return jsonify({
            "error": (
                "Unsupported format. Use MP4, MOV, "
                "AVI, MKV or WEBM."
            )
        }), 400

    safe_name = f"{uuid.uuid4().hex}{suffix}"

    saved_path = UPLOAD_DIR / safe_name

    uploaded.save(saved_path)

    try:

        info = inspect_video(
            str(saved_path),
            max_frames=120,
            sample_every=10
        )

        total_faces = 0

        # Face detection is separate from CNN classification
        for _, frame in info["sampled_frames"]:

            faces = face_detector.detect(frame)

            total_faces += len(faces)

        # CNN classification uses complete frames
        prediction = classify_video_frames(
            info["sampled_frames"]
        )

        result = {
            "filename": uploaded.filename,
            "duration_seconds": info["duration_seconds"],
            "fps": info["fps"],
            "resolution": (
                f'{info["width"]} × {info["height"]}'
            ),
            "total_frames": info["total_frames"],
            "sampled_frames": len(info["sampled_frames"]),
            "faces_detected_across_samples": total_faces,

            "classification": prediction["classification"],
            "confidence": prediction["confidence"],
            "real_probability": prediction["real_probability"],
            "fake_probability": prediction["fake_probability"],
            "predicted_frames": prediction["predicted_frames"],

            "status": prediction["status"],
            "message": prediction["message"]
        }

        add_analysis(
            uploaded.filename,
            info["duration_seconds"],
            len(info["sampled_frames"]),
            total_faces,
            result["status"]
        )

        return jsonify(result)

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 422


@app.get("/api/history")
def history():

    return jsonify(get_history())


@app.post("/api/analyze-frame")
def analyze_frame():

    image_file = request.files.get("frame")

    if not image_file:
        return jsonify({
            "error": "No webcam frame received."
        }), 400

    data = image_file.read()

    arr = np.frombuffer(
        data,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return jsonify({
            "error": "Could not decode webcam frame."
        }), 400

    # Detect faces
    faces = face_detector.detect(frame)

    # Classify the complete webcam frame
    prediction = deepfake_detector.predict(frame)

    return jsonify({
        "faces_detected": int(len(faces)),
        "classification": prediction["label"],
        "confidence": prediction["confidence"],
        "real_probability": prediction["real_probability"],
        "fake_probability": prediction["fake_probability"],
        "status": prediction["status"],
        "message": prediction["message"]
    })


# ==========================================
# WEBRTC VIDEO CALL SIGNALING
# ==========================================

rooms = {}
rooms_lock = threading.Lock()


@app.post("/api/rooms/create")
def create_room():
    """Create a new video call room."""

    room_id = secrets.token_urlsafe(6).replace("-", "").replace("_", "")

    with rooms_lock:
        rooms[room_id] = {
            "guest": False,
            "signals": [],
            "created_at": time.time()
        }

    return jsonify({
        "success": True,
        "room_id": room_id,
        "role": "host"
    })


@app.post("/api/rooms/<room_id>/join")
def join_room(room_id):
    """Allow a second user to join an existing room."""

    with rooms_lock:
        room = rooms.get(room_id)

        if room is None:
            return jsonify({
                "error": "Room not found."
            }), 404

        if room["guest"]:
            return jsonify({
                "error": "Room is already full."
            }), 409

        room["guest"] = True

    return jsonify({
        "success": True,
        "room_id": room_id,
        "role": "guest"
    })


@app.get("/api/rooms/<room_id>/status")
def room_status(room_id):
    """Check whether the second user has joined."""

    with rooms_lock:
        room = rooms.get(room_id)

        if room is None:
            return jsonify({"error": "Room not found."}), 404

        return jsonify({
            "exists": True,
            "guest_joined": room["guest"]
        })


@app.post("/api/rooms/<room_id>/signal")
def send_signal(room_id):
    """Exchange WebRTC offer, answer and ICE candidates."""

    data = request.get_json(silent=True) or {}

    role = data.get("role")
    signal_type = data.get("type")
    signal_data = data.get("data")

    if role not in ("host", "guest"):
        return jsonify({"error": "Invalid role."}), 400

    if signal_type not in ("offer", "answer", "candidate"):
        return jsonify({"error": "Invalid signal type."}), 400

    with rooms_lock:
        room = rooms.get(room_id)

        if room is None:
            return jsonify({"error": "Room not found."}), 404

        if role == "guest" and not room["guest"]:
            return jsonify({"error": "Guest has not joined."}), 403

        room["signals"].append({
            "from": role,
            "type": signal_type,
            "data": signal_data
        })

    return jsonify({"success": True})


@app.get("/api/rooms/<room_id>/signals")
def get_signals(room_id):
    """Read signaling messages sent by the other user."""

    role = request.args.get("role")
    since = request.args.get("since", default=0, type=int)

    if role not in ("host", "guest"):
        return jsonify({"error": "Invalid role."}), 400

    with rooms_lock:
        room = rooms.get(room_id)

        if room is None:
            return jsonify({"error": "Room not found."}), 404

        messages = [
            signal
            for signal in room["signals"][since:]
            if signal["from"] != role
        ]

        next_cursor = len(room["signals"])

    return jsonify({
        "messages": messages,
        "next": next_cursor
    })


@app.post("/api/rooms/<room_id>/leave")
def leave_room(room_id):
    """Remove a room when its host ends the call."""

    data = request.get_json(silent=True) or {}
    role = data.get("role")

    if role not in ("host", "guest"):
        return jsonify({"error": "Invalid role."}), 400

    with rooms_lock:
        room = rooms.get(room_id)

        if room is None:
            return jsonify({"success": True})

        if role == "host":
            rooms.pop(room_id, None)
        else:
            room["guest"] = False

    return jsonify({"success": True})

if __name__ == "__main__":

    print(
        "Starting Deepfake Detection System "
        "on http://127.0.0.1:5000"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
