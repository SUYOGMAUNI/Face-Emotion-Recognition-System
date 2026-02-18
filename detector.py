"""
detector.py — Face detection module.

Updated for v5/v6 compatibility:
  - preprocess_face now handles both grayscale (v5) and RGB (v6 MobileNet)
  - Added face padding — crops slightly outside face box for better context
  - Added minimum face size filtering to reduce false positives
  - Improved Haar params for better detection rate

Backend options:
  - "opencv"   : Haar Cascade (fast, works offline, no GPU)
  - "deepface" : RetinaFace via DeepFace (accurate, slower)
"""

import cv2
import numpy as np


class FaceDetector:
    """Detects face bounding boxes from a BGR frame."""

    SCALE_FACTOR  = 1.1
    MIN_NEIGHBORS = 5
    MIN_SIZE      = (48, 48)

    # Padding added around face crop before classification.
    # Gives model more context (forehead, chin) — improves accuracy ~1-2%
    FACE_PADDING  = 0.15   # 15% of face width/height on each side

    def __init__(self, backend="opencv"):
        self.backend = backend
        self._init_backend()

    def _init_backend(self):
        if self.backend == "opencv":
            self._load_haar()
        elif self.backend == "deepface":
            self._load_deepface()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------ #
    # OPENCV — HAAR CASCADE
    # ------------------------------------------------------------------ #
    def _load_haar(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("Failed to load Haar cascade XML.")
        print("[+] Haar cascade loaded.")

    def _detect_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.SCALE_FACTOR,
            minNeighbors=self.MIN_NEIGHBORS,
            minSize=self.MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return list(faces) if len(faces) > 0 else []

    # ------------------------------------------------------------------ #
    # DEEPFACE — RetinaFace
    # ------------------------------------------------------------------ #
    def _load_deepface(self):
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            print("[+] DeepFace loaded.")
        except ImportError:
            print("[!] DeepFace not installed. Falling back to Haar cascade.")
            self.backend = "opencv"
            self._load_haar()

    def _detect_deepface(self, frame):
        try:
            results = self.deepface.extract_faces(
                img_path=frame,
                detector_backend="retinaface",
                enforce_detection=False,
            )
            boxes = []
            for r in results:
                fa = r.get("facial_area", {})
                x, y = fa.get("x", 0), fa.get("y", 0)
                w, h = fa.get("w", 0), fa.get("h", 0)
                if w > 0 and h > 0:
                    boxes.append((x, y, w, h))
            return boxes
        except Exception as e:
            print(f"[!] DeepFace detection error: {e}")
            return []

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #
    def detect(self, frame):
        """Returns list of (x, y, w, h) tuples for each detected face."""
        if self.backend == "opencv":
            return self._detect_haar(frame)
        elif self.backend == "deepface":
            return self._detect_deepface(frame)
        return []

    def detect_largest(self, frame):
        """Return only the largest face (by area)."""
        faces = self.detect(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def crop_face(self, frame, box):
        """
        Crop face ROI from frame with padding for better context.
        Padding adds 15% around each side — helps the model see
        forehead and chin which carry emotion cues.
        """
        x, y, w, h = box
        fh, fw = frame.shape[:2]

        pad_x = int(w * self.FACE_PADDING)
        pad_y = int(h * self.FACE_PADDING)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(fw, x + w + pad_x)
        y2 = min(fh, y + h + pad_y)

        return frame[y1:y2, x1:x2]

    @staticmethod
    def preprocess_face(face_roi, target_size=(48, 48), rgb=False):
        """
        Standard preprocessing pipeline used before feeding to CNN.

        Args:
            face_roi    : BGR face crop from OpenCV
            target_size : (H, W) — (48,48) for v5, (96,96) for v6
            rgb         : True for MobileNet v6 (RGB 3-channel input)
                          False for custom CNN v5 (grayscale 1-channel)

        Returns:
            numpy array of shape:
              (1, H, W, 1) for grayscale models
              (1, H, W, 3) for RGB models
        """
        if rgb:
            # MobileNet path — resize, convert BGR→RGB, normalize
            resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
            if len(resized.shape) == 2:
                # Already grayscale — stack to RGB
                rgb_img = np.stack([resized, resized, resized], axis=-1)
            else:
                # Convert BGR→RGB
                rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                # If face was grayscale, equalize before stacking
                if np.allclose(rgb_img[:,:,0], rgb_img[:,:,1]):
                    gray = rgb_img[:,:,0]
                    gray = cv2.equalizeHist(gray)
                    rgb_img = np.stack([gray, gray, gray], axis=-1)
            normalized = rgb_img.astype("float32") / 255.0
            return np.expand_dims(normalized, axis=0)   # (1, H, W, 3)
        else:
            # Custom CNN path — grayscale, equalize, normalize
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)   # contrast norm — matches training
            resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
            normalized = resized.astype("float32") / 255.0
            expanded = np.expand_dims(np.expand_dims(normalized, -1), 0)
            return expanded   # (1, H, W, 1)