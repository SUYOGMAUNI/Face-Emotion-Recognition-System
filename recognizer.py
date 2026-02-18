"""
recognizer.py — Core emotion recognition engine.

Updated for v5/v6:
  - Uses crop_face() with padding for better face context
  - Passes rgb flag to preprocess_face based on model type
  - FPS-based frame skipping — only classify every N frames (reduces lag)
  - Cached last emotion per face — smoother display between classifications
  - Screenshot saves clean frame without HUD noise
"""

import cv2
import time
import datetime
import numpy as np
from collections import defaultdict, deque
from detector import FaceDetector
from classifier import EmotionClassifier
from overlay import draw_overlay


class EmotionRecognizer:
    """
    Orchestrates the full pipeline:
      capture → detect faces → classify emotion → draw overlay → display/save
    """

    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    EMOJI_MAP = {
        "Angry":    "😠",
        "Disgust":  "🤢",
        "Fear":     "😨",
        "Happy":    "😊",
        "Neutral":  "😐",
        "Sad":      "😢",
        "Surprise": "😲",
    }

    # Only run classifier every N frames — reduces CPU/GPU load
    # Set to 1 to classify every frame (slower but more responsive)
    CLASSIFY_EVERY_N = 3

    def __init__(
        self,
        backend="opencv",
        save_output=False,
        output_path="output.avi",
        display=True,
        collect_stats=True,
    ):
        self.backend       = backend
        self.save_output   = save_output
        self.output_path   = output_path
        self.display       = display
        self.collect_stats = collect_stats

        self.detector   = FaceDetector(backend=backend)
        self.classifier = EmotionClassifier(backend=backend)

        # Stats
        self.emotion_counts = defaultdict(int)
        self.frame_count    = 0
        self.face_count     = 0
        self.fps_history    = deque(maxlen=30)
        self.start_time     = None
        self.screenshot_idx = 0

        # Cache last classification result per face position
        # key = rough face position hash, value = (emotion, scores)
        self._last_results  = {}

        self.writer = None

    # ------------------------------------------------------------------ #
    # MAIN LOOP
    # ------------------------------------------------------------------ #
    def run(self, source):
        self.start_time = time.time()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[!] Cannot open source: {source}")
            return

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[*] Stream ended.")
                break

            self.frame_count += 1
            result_frame = self._process_frame(frame)

            # FPS
            elapsed = time.time() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)

            # Draw HUD
            result_frame = draw_overlay(
                result_frame,
                fps=avg_fps,
                frame_num=self.frame_count,
                face_count=self.face_count,
                emotion_counts=self.emotion_counts,
                backend=self.backend,
                session_time=time.time() - self.start_time,
            )

            if self.save_output:
                self._write_frame(result_frame)

            if self.display:
                cv2.imshow("Emotion Recognizer — Suyog Mauni", result_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    # Save screenshot of raw frame (no HUD)
                    self._save_screenshot(frame)
                elif key == ord("r"):
                    self._reset_stats()

        cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    # FRAME PROCESSING
    # ------------------------------------------------------------------ #
    def _process_frame(self, frame):
        """Detect faces → classify emotion → annotate frame."""
        faces = self.detector.detect(frame)
        self.face_count = len(faces)

        # Only classify every N frames — use cached result otherwise
        do_classify = (self.frame_count % self.CLASSIFY_EVERY_N == 0)

        for (x, y, w, h) in faces:
            # Use padded crop for classification (more context)
            face_roi = self.detector.crop_face(frame, (x, y, w, h))
            if face_roi.size == 0:
                continue

            # Face position hash — rough identity for caching
            face_key = (x // 20, y // 20)

            if do_classify or face_key not in self._last_results:
                emotion, scores = self.classifier.classify(face_roi)
                self._last_results[face_key] = (emotion, scores)

                if self.collect_stats and emotion:
                    self.emotion_counts[emotion] += 1
            else:
                # Use cached result from previous frame
                emotion, scores = self._last_results[face_key]

            frame = self._draw_face(frame, x, y, w, h, emotion, scores)

        # Clean up stale face cache entries
        if len(self._last_results) > 20:
            self._last_results.clear()

        return frame

    def _draw_face(self, frame, x, y, w, h, emotion, scores):
        """Draw bounding box, emotion label, and confidence bars."""
        COLOR_MAP = {
            "Happy":    (0, 220, 100),
            "Sad":      (200, 100, 50),
            "Angry":    (50, 50, 220),
            "Surprise": (50, 200, 220),
            "Fear":     (150, 50, 200),
            "Disgust":  (50, 180, 80),
            "Neutral":  (180, 180, 180),
        }
        color = COLOR_MAP.get(emotion, (200, 200, 200))

        # Bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Corner accents
        corner, thick = 15, 2
        for cx, cy, dx, dy in [
            (x,   y,   1,  1), (x+w, y,   -1,  1),
            (x,   y+h, 1, -1), (x+w, y+h, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx+dx*corner, cy), color, thick+1)
            cv2.line(frame, (cx, cy), (cx, cy+dy*corner), color, thick+1)

        # Label
        conf = scores.get(emotion, 0) if scores else 0
        label = f"{emotion}  {conf:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        cv2.rectangle(frame, (x, y-th-14), (x+tw+10, y), color, -1)
        cv2.putText(frame, label, (x+5, y-6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (10, 10, 10), 1, cv2.LINE_AA)

        # Confidence bars for all 7 emotions (sorted by score)
        if scores:
            bar_x, bar_y = x + w + 8, y
            bar_w, bar_h = 80, 10
            sorted_scores = sorted(scores.items(), key=lambda e: -e[1])
            for i, (emo, score) in enumerate(sorted_scores):
                emo_color = COLOR_MAP.get(emo, (160, 160, 160))
                filled = int(bar_w * score / 100)
                by = bar_y + i * (bar_h + 4)
                if by + bar_h >= frame.shape[0]:
                    break
                cv2.rectangle(frame, (bar_x, by), (bar_x+bar_w, by+bar_h),
                              (40, 40, 40), -1)
                if filled > 0:
                    cv2.rectangle(frame, (bar_x, by), (bar_x+filled, by+bar_h),
                                  emo_color, -1)
                cv2.putText(frame, f"{emo[:3]} {score:.0f}%",
                            (bar_x+bar_w+4, by+bar_h-2),
                            cv2.FONT_HERSHEY_PLAIN, 0.7, emo_color, 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #
    def _write_frame(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, 20.0, (w, h))
        self.writer.write(frame)

    def _save_screenshot(self, frame):
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"screenshot_{ts}_{self.screenshot_idx}.jpg"
        cv2.imwrite(path, frame)
        self.screenshot_idx += 1
        print(f"[+] Screenshot saved: {path}")

    def _reset_stats(self):
        self.emotion_counts.clear()
        self.frame_count = 0
        self.face_count  = 0
        self.start_time  = time.time()
        self._last_results.clear()
        print("[*] Stats reset.")

    def print_stats(self):
        total    = sum(self.emotion_counts.values()) or 1
        duration = time.time() - (self.start_time or time.time())
        print("\n" + "="*45)
        print("  SESSION EMOTION STATISTICS")
        print("="*45)
        print(f"  Duration    : {duration:.1f}s")
        print(f"  Frames      : {self.frame_count}")
        print(f"  Avg FPS     : {self.frame_count / duration:.1f}")
        print(f"  Total Faces : {sum(self.emotion_counts.values())}")
        print("-"*45)
        for emo in self.EMOTIONS:
            count = self.emotion_counts.get(emo, 0)
            pct   = count / total * 100
            bar   = "█" * int(pct / 5)
            emoji = self.EMOJI_MAP.get(emo, "")
            print(f"  {emoji} {emo:<10} {bar:<20} {pct:5.1f}%  ({count})")
        print("="*45 + "\n")