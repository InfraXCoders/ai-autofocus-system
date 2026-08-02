"""AI Cinema Autofocus System — live desktop demo entry point.

Opens your webcam, runs the full autofocus pipeline, and draws an overlay:
  * green box  = tracked subject (the focus target)
  * cyan boxes = other detected faces
  * yellow     = eyes
  * magenta    = body/person
  * HUD        = distance estimate, focus-ring position, lens, FPS

Click a face to lock focus onto it (tap-to-track).

Lens calibration (Phase 3):
    l           cycle the active lens
    c           toggle calibration mode (pauses auto-focus)
    i / k       while calibrating: rack focus in / out by hand
    a           while calibrating: save a calibration point at the
                subject's current estimated distance
    s           save the active lens's calibration to disk
    q           quit

Usage:
    python main.py                    # webcam 0
    python main.py --source 1         # another webcam
    python main.py --source clip.mp4  # a video file
    python main.py --no-mesh          # faster (skip detailed eye landmarks)

Want to let someone else test with THEIR OWN camera over the internet
(e.g. via ngrok)? Run `python web_main.py` instead — see its docstring.
"""

import argparse
import time

import cv2

import config
from src.overlay import draw_hud, draw_overlay, handle_key
from src.pipeline import AutofocusPipeline


def parse_args():
    p = argparse.ArgumentParser(description="AI Autofocus — live vision demo")
    p.add_argument("--source", default="0",
                   help="Webcam index (e.g. 0) or path to a video file")
    p.add_argument("--no-mesh", action="store_true",
                   help="Disable detailed eye-mesh detection (faster)")
    p.add_argument("--no-pose", action="store_true",
                   help="Disable body/pose detection (faster)")
    return p.parse_args()


def open_capture(source: str):
    # Numeric source -> webcam index; otherwise treat as a file path.
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(
            f"[error] Could not open video source '{source}'. "
            "Check the webcam index or file path, and grant camera permission."
        )
    return cap


def main():
    args = parse_args()
    if args.no_mesh:
        config.ENABLE_FACE_MESH = False
    if args.no_pose:
        config.ENABLE_POSE = False

    pipeline = AutofocusPipeline()
    cap = open_capture(args.source)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pipeline.select_at(x, y)

    cv2.namedWindow(config.WINDOW_NAME)
    cv2.setMouseCallback(config.WINDOW_NAME, on_mouse)

    prev_t = time.time()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror for natural webcam feel

            state = pipeline.step(frame)

            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            draw_overlay(frame, state)
            draw_hud(frame, state, fps, draw_fps=config.DRAW_FPS)

            cv2.imshow(config.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            handle_key(pipeline, chr(key) if 32 <= key < 127 else "",
                       state["distance_m"], config.CALIBRATION_STEP)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()


if __name__ == "__main__":
    main()
