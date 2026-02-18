"""
app.py — Web UI for Face Emotion Recognition
Author: Suyog Mauni | suyogmauni.com.np

Flask-based web interface with real-time video streaming.
Replaces OpenCV window with browser-based UI.

Usage:
    python app.py
    Then open: http://localhost:5000
"""

import cv2
import time
import json
import threading
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from collections import defaultdict, deque
from detector import FaceDetector
from classifier import EmotionClassifier

app = Flask(__name__)
CORS(app)

# Global state
camera = None
camera_lock = threading.Lock()
detector = None
classifier = None
stats = {
    "emotion_counts": defaultdict(int),
    "frame_count": 0,
    "face_count": 0,
    "fps": 0,
    "session_start": time.time(),
    "is_running": False
}
stats_lock = threading.Lock()
fps_history = deque(maxlen=30)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_COLORS = {
    "Happy": "#64DC00",
    "Sad": "#C86432",
    "Angry": "#DC3232",
    "Surprise": "#DCC832",
    "Fear": "#C832C8",
    "Disgust": "#50B450",
    "Neutral": "#B4B4B4",
}


def init_models(backend="opencv"):
    """Initialize detector and classifier."""
    global detector, classifier
    detector = FaceDetector(backend=backend)
    classifier = EmotionClassifier(backend=backend)
    print(f"[+] Models initialized with backend: {backend}")


def process_frame(frame):
    """Process a single frame and return annotated result."""
    global stats, fps_history
    
    t_start = time.time()
    
    # Detect faces
    faces = detector.detect(frame)
    
    with stats_lock:
        stats["face_count"] = len(faces)
        stats["frame_count"] += 1
    
    # Process each face
    for (x, y, w, h) in faces:
        face_roi = detector.crop_face(frame, (x, y, w, h))
        if face_roi.size == 0:
            continue
        
        # Classify emotion
        emotion, scores = classifier.classify(face_roi)
        
        # Update stats
        with stats_lock:
            if emotion:
                stats["emotion_counts"][emotion] += 1
        
        # Draw bounding box and label
        frame = draw_face(frame, x, y, w, h, emotion, scores)
    
    # Calculate FPS
    elapsed = time.time() - t_start
    fps = 1.0 / elapsed if elapsed > 0 else 0
    fps_history.append(fps)
    
    with stats_lock:
        stats["fps"] = sum(fps_history) / len(fps_history)
    
    return frame


def draw_face(frame, x, y, w, h, emotion, scores):
    """Draw bounding box and emotion label on face."""
    color_map = {
        "Happy": (0, 220, 100),
        "Sad": (200, 100, 50),
        "Angry": (50, 50, 220),
        "Surprise": (50, 200, 220),
        "Fear": (150, 50, 200),
        "Disgust": (50, 180, 80),
        "Neutral": (180, 180, 180),
    }
    color = color_map.get(emotion, (200, 200, 200))
    
    # Bounding box
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Corner accents
    corner_len = 15
    thick = 2
    for cx, cy, dx, dy in [
        (x, y, 1, 1), (x+w, y, -1, 1),
        (x, y+h, 1, -1), (x+w, y+h, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx+dx*corner_len, cy), color, thick+1)
        cv2.line(frame, (cx, cy), (cx, cy+dy*corner_len), color, thick+1)
    
    # Label background
    conf = scores.get(emotion, 0) if scores else 0
    label = f"{emotion}  {conf:.0f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    cv2.rectangle(frame, (x, y-th-14), (x+tw+10, y), color, -1)
    cv2.putText(frame, label, (x+5, y-6),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (10, 10, 10), 1, cv2.LINE_AA)
    
    return frame


def generate_frames():
    """Generator function for video streaming."""
    global camera, stats
    
    with stats_lock:
        stats["is_running"] = True
    
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            # Give camera time to warm up
            time.sleep(0.5)
    
    try:
        while True:
            with camera_lock:
                if camera is None or not camera.isOpened():
                    break
                success, frame = camera.read()
            
            if not success:
                break
            
            # Process frame
            frame = process_frame(frame)
            
            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        with stats_lock:
            stats["is_running"] = False


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def get_stats():
    """Return current statistics as JSON."""
    with stats_lock:
        total = sum(stats["emotion_counts"].values()) or 1
        emotion_percentages = {
            emotion: {
                "count": stats["emotion_counts"].get(emotion, 0),
                "percentage": (stats["emotion_counts"].get(emotion, 0) / total) * 100,
                "color": EMOTION_COLORS.get(emotion, "#999999")
            }
            for emotion in EMOTIONS
        }
        
        session_time = time.time() - stats["session_start"]
        
        return jsonify({
            "emotion_data": emotion_percentages,
            "frame_count": stats["frame_count"],
            "face_count": stats["face_count"],
            "fps": round(stats["fps"], 1),
            "session_time": round(session_time, 1),
            "is_running": stats["is_running"]
        })


@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    """Reset statistics."""
    global stats, fps_history
    with stats_lock:
        stats["emotion_counts"].clear()
        stats["frame_count"] = 0
        stats["face_count"] = 0
        stats["session_start"] = time.time()
        fps_history.clear()
    return jsonify({"status": "success"})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop the camera."""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    with stats_lock:
        stats["is_running"] = False
    return jsonify({"status": "stopped"})


@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start the camera."""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            time.sleep(0.5)
    return jsonify({"status": "started"})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🎭 Face Emotion Recognition - Web UI")
    print("  Author: Suyog Mauni | suyogmauni.com.np")
    print("="*60)
    print("\n[*] Initializing models...")
    
    init_models(backend="opencv")
    
    print("[+] Server starting...")
    print("[+] Open your browser and navigate to:")
    print("    → http://localhost:5000")
    print("    → http://127.0.0.1:5000")
    print("\n[*] Press Ctrl+C to stop the server\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()