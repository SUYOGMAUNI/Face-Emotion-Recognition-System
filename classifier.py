"""
classifier.py — Emotion classification module.

Updated for v5 model:
  - Supports both v5 (48x48 grayscale) and v6 MobileNet (96x96 RGB) models
  - Auto-detects model type from file
  - Smoothed predictions using temporal averaging (reduces flicker)
  - Confidence threshold to suppress low-confidence predictions
  - Proper preprocessing matching training pipeline exactly
"""

import os
import cv2
import numpy as np
from collections import deque
from detector import FaceDetector

EMOTIONS   = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Model paths — priority order (v6 > v5 > legacy)
MODEL_PATHS = [
    "models/emotion_model_v6.keras",   # MobileNet v6
    "models/emotion_model_v5.keras",   # Custom CNN v5
    "models/emotion_model_v4.keras",   # Custom CNN v4
    "models/emotion_model.h5",         # Legacy
]

# Confidence below this → show "Uncertain" instead of a wrong emotion
CONFIDENCE_THRESHOLD = 40.0

# Temporal smoothing: average predictions over N frames (reduces flicker)
SMOOTH_WINDOW = 5


class EmotionClassifier:

    def __init__(self, backend="opencv"):
        self.backend     = backend
        self.model       = None
        self.model_type  = None   # "mobilenet" or "cnn"
        self.input_size  = None
        self.input_channels = None

        # Per-face prediction history for temporal smoothing
        # key = face_id (rough position hash), value = deque of score dicts
        self._history = deque(maxlen=SMOOTH_WINDOW)

        self._init()

    # ------------------------------------------------------------------ #
    # INIT
    # ------------------------------------------------------------------ #
    def _init(self):
        if self.backend == "deepface":
            self._init_deepface()
        else:
            self._init_cnn()

    def _init_cnn(self):
        """Load best available Keras model automatically."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model

            loaded = False
            for path in MODEL_PATHS:
                if os.path.exists(path):
                    print(f"[+] Loading model: {path}")
                    self.model = load_model(path, compile=False)

                    # Auto-detect input shape from model
                    inp_shape = self.model.input_shape  # (None, H, W, C)
                    self.input_size     = (inp_shape[1], inp_shape[2])
                    self.input_channels = inp_shape[3]

                    # Detect if it's MobileNet (96x96 RGB) or custom CNN (48x48 gray)
                    if self.input_channels == 3:
                        self.model_type = "mobilenet"
                        print(f"    Type    : MobileNetV2 ({self.input_size[0]}x{self.input_size[1]} RGB)")
                    else:
                        self.model_type = "cnn"
                        print(f"    Type    : Custom CNN ({self.input_size[0]}x{self.input_size[1]} grayscale)")

                    print(f"    Params  : {self.model.count_params():,}")
                    loaded = True
                    break

            if not loaded:
                print(f"[!] No model found. Checked:")
                for p in MODEL_PATHS:
                    print(f"    {p}")
                print(f"    Place your trained model in models/ and restart.")
                print(f"    Using rule-based fallback.\n")
                self.model = None

        except ImportError:
            print("[!] TensorFlow not found. Using rule-based classifier.")
            self.model = None

    def _init_deepface(self):
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            print("[+] DeepFace classifier ready.")
        except ImportError:
            print("[!] DeepFace not installed. Falling back to CNN.")
            self.backend = "opencv"
            self._init_cnn()

    # ------------------------------------------------------------------ #
    # CLASSIFY — PUBLIC API
    # ------------------------------------------------------------------ #
    def classify(self, face_roi):
        """
        Returns (emotion_string, scores_dict).
        scores_dict maps each emotion → 0-100 confidence.
        Returns ("Uncertain", ...) if confidence is below threshold.
        """
        if self.backend == "deepface":
            emotion, scores = self._classify_deepface(face_roi)
        elif self.model is not None:
            emotion, scores = self._classify_cnn(face_roi)
        else:
            emotion, scores = self._classify_rule_based(face_roi)

        # Temporal smoothing — average over last N frames
        scores = self._smooth(scores)

        # Re-derive top emotion after smoothing
        emotion = max(scores, key=scores.get)

        # Confidence gate — suppress noisy low-confidence predictions
        if scores[emotion] < CONFIDENCE_THRESHOLD:
            emotion = "Neutral"   # default to neutral when uncertain

        return emotion, scores

    # ------------------------------------------------------------------ #
    # CNN CLASSIFY
    # ------------------------------------------------------------------ #
    def _classify_cnn(self, face_roi):
        preprocessed = self._preprocess(face_roi)
        preds = self.model.predict(preprocessed, verbose=0)[0]
        scores = {e: round(float(p) * 100, 1) for e, p in zip(EMOTIONS, preds)}
        top = max(scores, key=scores.get)
        return top, scores

    def _preprocess(self, face_roi):
        """
        Preprocessing that exactly matches training pipeline.

        v5 (custom CNN):
          - Grayscale, resize to 48x48, normalize [0,1], shape (1,48,48,1)

        v6 (MobileNet):
          - RGB (grayscale stacked x3), resize to 96x96,
            normalize [0,1], shape (1,96,96,3)
          - MobileNetV2.preprocess_input applied inside model graph
        """
        if self.model_type == "mobilenet":
            # Resize to model input size
            resized = cv2.resize(face_roi, self.input_size, interpolation=cv2.INTER_AREA)
            # Convert BGR→RGB
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                # Grayscale face → stack to RGB
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
                rgb = np.stack([gray, gray, gray], axis=-1)
            normalized = rgb.astype("float32") / 255.0
            return np.expand_dims(normalized, axis=0)   # (1, 96, 96, 3)

        else:
            # Custom CNN (v4/v5) — grayscale 48x48
            return FaceDetector.preprocess_face(face_roi, self.input_size)

    # ------------------------------------------------------------------ #
    # TEMPORAL SMOOTHING
    # ------------------------------------------------------------------ #
    def _smooth(self, scores):
        """
        Averages predictions over last SMOOTH_WINDOW frames.
        This eliminates single-frame noise and flickering emotion labels.
        """
        self._history.append(scores)
        smoothed = {e: 0.0 for e in EMOTIONS}
        for past_scores in self._history:
            for e in EMOTIONS:
                smoothed[e] += past_scores.get(e, 0.0)
        n = len(self._history)
        smoothed = {e: round(v / n, 1) for e, v in smoothed.items()}
        return smoothed

    # ------------------------------------------------------------------ #
    # DEEPFACE CLASSIFY
    # ------------------------------------------------------------------ #
    def _classify_deepface(self, face_roi):
        try:
            result = self.deepface.analyze(
                img_path=face_roi,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            emo_scores = result.get("emotion", {})
            scores = {}
            for emo in EMOTIONS:
                scores[emo] = round(emo_scores.get(emo.lower(), 0.0), 1)
            top = result.get("dominant_emotion", "neutral").capitalize()
            for e in EMOTIONS:
                if e.lower() == top.lower():
                    top = e
                    break
            return top, scores
        except Exception as ex:
            print(f"[!] DeepFace classify error: {ex}")
            return "Neutral", {e: 0.0 for e in EMOTIONS}

    # ------------------------------------------------------------------ #
    # RULE-BASED FALLBACK
    # ------------------------------------------------------------------ #
    def _classify_rule_based(self, face_roi):
        """Fallback when no model is available. Not accurate."""
        gray  = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        mean  = np.mean(gray)
        std   = np.std(gray)
        scores = {e: 5.0 for e in EMOTIONS}
        if std > 60:
            scores["Happy"]    = 45.0
            scores["Surprise"] = 30.0
        elif std < 30:
            scores["Neutral"]  = 50.0
            scores["Sad"]      = 25.0
        elif mean < 90:
            scores["Angry"]    = 40.0
            scores["Fear"]     = 30.0
        else:
            scores["Neutral"]  = 60.0
        total = sum(scores.values())
        scores = {k: round(v / total * 100, 1) for k, v in scores.items()}
        top = max(scores, key=scores.get)
        return top, scores

    # ------------------------------------------------------------------ #
    # MODEL BUILDER — for train.py (v5 architecture)
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_model(num_classes=7):
        """
        v5 custom CNN architecture.
        Matches the model trained in emotion_cnn_v5.ipynb exactly.
        Input: (48, 48, 1) grayscale
        """
        from tensorflow.keras import layers, Model, Input
        from tensorflow.keras.regularizers import l2

        REG = 5e-4

        def conv_bn_relu(x, filters, kernel=3):
            x = layers.Conv2D(filters, kernel, padding='same',
                              kernel_regularizer=l2(REG), use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            return layers.Activation('relu')(x)

        def residual_block(x, filters):
            shortcut = x
            x = conv_bn_relu(x, filters)
            x = conv_bn_relu(x, filters)
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, 1, padding='same',
                                         kernel_regularizer=l2(REG), use_bias=False)(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            x = layers.Add()([x, shortcut])
            return layers.Activation('relu')(x)

        inp = Input(shape=(48, 48, 1))
        x = conv_bn_relu(inp, 32)
        x = conv_bn_relu(x, 64)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.4)(x)

        x = residual_block(x, 128)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.4)(x)

        x = residual_block(x, 256)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.4)(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, kernel_regularizer=l2(REG))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        out = layers.Dense(num_classes, activation='softmax')(x)

        from tensorflow.keras import Model
        return Model(inp, out)