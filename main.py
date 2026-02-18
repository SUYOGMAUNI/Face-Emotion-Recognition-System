"""
Emotion Recognizer — main.py
Author: Suyog Mauni | suyogmauni.com.np

Entry point. Choose between:
  - webcam mode (real-time)
  - image mode  (single file)
  - video mode  (video file)

Usage:
  python main.py                        # webcam (default)
  python main.py --source image.jpg     # image file
  python main.py --source video.mp4     # video file
  python main.py --source 0 --save      # webcam + save output
  python main.py --stats                # show emotion stats after session

Model setup:
  Place your trained model in the models/ folder:
    models/emotion_model_v6.keras   <- MobileNet v6 (best)
    models/emotion_model_v5.keras   <- Custom CNN v5
  The classifier auto-detects which model to load.
"""

import argparse
from recognizer import EmotionRecognizer


def parse_args():
    p = argparse.ArgumentParser(description="Real-Time Face Emotion Recognition")
    p.add_argument("--source",  default="0",
                   help="0 = webcam, or path to image/video file")
    p.add_argument("--backend", default="opencv",
                   choices=["opencv", "deepface"],
                   help="Detection backend (opencv=fast, deepface=accurate)")
    p.add_argument("--save",    action="store_true",
                   help="Save output to file")
    p.add_argument("--output",  default="output.avi",
                   help="Output file path when --save is used")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless (no window) — useful for servers")
    p.add_argument("--stats",   action="store_true",
                   help="Print emotion statistics after session ends")
    p.add_argument("--classify-every", type=int, default=3,
                   help="Classify every N frames (default=3, lower=smoother but slower)")
    return p.parse_args()


def main():
    args = parse_args()

    source = args.source
    if source.isdigit():
        source = int(source)

    recognizer = EmotionRecognizer(
        backend=args.backend,
        save_output=args.save,
        output_path=args.output,
        display=not args.no_display,
        collect_stats=args.stats,
    )

    # Override classify frequency if specified
    recognizer.CLASSIFY_EVERY_N = args.classify_every

    print(f"[*] Starting Emotion Recognizer")
    print(f"    Source         : {source}")
    print(f"    Backend        : {args.backend}")
    print(f"    Save           : {args.save}")
    print(f"    Classify every : {args.classify_every} frames")
    print(f"[*] Press 'q' to quit | 's' to screenshot | 'r' to reset stats\n")

    recognizer.run(source)

    if args.stats:
        recognizer.print_stats()


if __name__ == "__main__":
    main()