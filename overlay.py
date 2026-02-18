"""
overlay.py — HUD (Heads-Up Display) drawn on each video frame.

Draws:
  - Top bar  : title, FPS, frame count, backend
  - Side bar : live emotion bar chart
  - Bottom   : session time, face count
"""

import cv2
import numpy as np

# BGR colour palette
GOLD    = (55, 175, 212)    # #D4AF37 in BGR
WHITE   = (230, 230, 230)
GREY    = (120, 120, 120)
BLACK   = (10,  10,  10)
GREEN   = (100, 220, 0)
RED     = (50,  50,  220)

EMOTION_COLORS_BGR = {
    "Happy":    (100, 220, 0),
    "Sad":      (50, 100, 200),
    "Angry":    (50,  50, 220),
    "Surprise": (220, 200, 50),
    "Fear":     (200, 50, 150),
    "Disgust":  (80, 180, 50),
    "Neutral":  (180, 180, 180),
}

FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN


def draw_overlay(frame, fps, frame_num, face_count,
                 emotion_counts, backend, session_time):
    """
    Draws the full HUD overlay on frame (in-place + returns frame).
    """
    h, w = frame.shape[:2]

    frame = _draw_top_bar(frame, w, fps, frame_num, backend)
    frame = _draw_bottom_bar(frame, w, h, face_count, session_time)
    frame = _draw_emotion_chart(frame, w, h, emotion_counts)

    return frame


# ------------------------------------------------------------------ #
# TOP BAR
# ------------------------------------------------------------------ #
def _draw_top_bar(frame, w, fps, frame_num, backend):
    bar_h = 38
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Gold accent line
    cv2.line(frame, (0, bar_h), (w, bar_h), GOLD, 1)

    # Title
    cv2.putText(frame, "EMOTION RECOGNIZER",
                (10, 26), FONT, 0.65, GOLD, 1, cv2.LINE_AA)

    # FPS — right aligned
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, FONT_MONO, 1.1, 1)
    fps_color = GREEN if fps >= 20 else (0, 165, 255) if fps >= 10 else RED
    cv2.putText(frame, fps_text,
                (w - tw - 120, 26), FONT_MONO, 1.1, fps_color, 1, cv2.LINE_AA)

    # Frame count
    frame_text = f"Frame: {frame_num}"
    cv2.putText(frame, frame_text,
                (w - 110, 26), FONT_MONO, 1.0, GREY, 1, cv2.LINE_AA)

    # Backend badge
    badge_text = f"[{backend.upper()}]"
    cv2.putText(frame, badge_text,
                (10, bar_h + 20), FONT_MONO, 0.9, GREY, 1, cv2.LINE_AA)

    return frame


# ------------------------------------------------------------------ #
# BOTTOM BAR
# ------------------------------------------------------------------ #
def _draw_bottom_bar(frame, w, h, face_count, session_time):
    bar_h = 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-bar_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.line(frame, (0, h-bar_h), (w, h-bar_h), GOLD, 1)

    mins = int(session_time // 60)
    secs = int(session_time % 60)
    time_str = f"Session: {mins:02d}:{secs:02d}"
    cv2.putText(frame, time_str,
                (10, h-10), FONT_MONO, 1.0, GREY, 1, cv2.LINE_AA)

    face_str = f"Faces Detected: {face_count}"
    face_color = GREEN if face_count > 0 else GREY
    cv2.putText(frame, face_str,
                (w//2 - 70, h-10), FONT_MONO, 1.0, face_color, 1, cv2.LINE_AA)

    hint = "q=quit  s=screenshot  r=reset"
    (tw, _), _ = cv2.getTextSize(hint, FONT_MONO, 0.85, 1)
    cv2.putText(frame, hint,
                (w - tw - 8, h-10), FONT_MONO, 0.85, GREY, 1, cv2.LINE_AA)

    return frame


# ------------------------------------------------------------------ #
# EMOTION BAR CHART (top-right corner)
# ------------------------------------------------------------------ #
def _draw_emotion_chart(frame, w, h, emotion_counts):
    if not emotion_counts:
        return frame

    emotions   = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
    total      = sum(emotion_counts.values()) or 1
    chart_w    = 160
    chart_h    = len(emotions) * 22 + 30
    margin     = 8
    cx         = w - chart_w - margin
    cy         = 60

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (cx-5, cy-5), (cx+chart_w+5, cy+chart_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    cv2.line(frame, (cx-5, cy-5), (cx+chart_w+5, cy-5), GOLD, 1)

    # Header
    cv2.putText(frame, "EMOTIONS",
                (cx, cy+14), FONT, 0.5, GOLD, 1, cv2.LINE_AA)

    bar_max_w = 80

    for i, emo in enumerate(emotions):
        count = emotion_counts.get(emo, 0)
        pct   = count / total
        color = EMOTION_COLORS_BGR.get(emo, GREY)

        y_pos = cy + 30 + i * 22

        # Emotion label
        cv2.putText(frame, f"{emo[:3]}",
                    (cx, y_pos + 10), FONT_MONO, 0.85, color, 1, cv2.LINE_AA)

        # Bar background
        cv2.rectangle(frame, (cx+32, y_pos), (cx+32+bar_max_w, y_pos+10),
                      (40, 40, 40), -1)

        # Bar fill
        filled = int(bar_max_w * pct)
        if filled > 0:
            cv2.rectangle(frame, (cx+32, y_pos),
                          (cx+32+filled, y_pos+10), color, -1)

        # Percentage text
        pct_text = f"{pct*100:.0f}%"
        cv2.putText(frame, pct_text,
                    (cx+32+bar_max_w+4, y_pos+10),
                    FONT_MONO, 0.8, color, 1, cv2.LINE_AA)

    return frame