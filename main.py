"""AI Cinema Autofocus System — live demo entry point (Phase 1).

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
"""

import argparse
import time

import cv2

import config
from src.geometry import iou_xywh
from src.pipeline import AutofocusPipeline

# Colors (BGR)
GREEN = (0, 220, 0)
CYAN = (220, 220, 0)
YELLOW = (0, 220, 220)
MAGENTA = (220, 0, 220)
WHITE = (240, 240, 240)
BLACK = (0, 0, 0)
AMBER = (0, 170, 255)


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


def draw_overlay(frame, state):
    subject = state["subject"]
    for d in state["detections"]:
        # Don't draw a faint box under the bold target box.
        if subject is not None and d.kind == subject.kind and \
                iou_xywh((d.x, d.y, d.w, d.h), (subject.x, subject.y, subject.w, subject.h)) > 0.5:
            continue
        color = {"face": CYAN, "eye": YELLOW, "body": MAGENTA}.get(d.kind, WHITE)
        cv2.rectangle(frame, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 1)

    # Tracked subject on top, in bold green.
    if subject is not None:
        cv2.rectangle(frame, (subject.x, subject.y),
                      (subject.x + subject.w, subject.y + subject.h), GREEN, 2)
        cx, cy = subject.center
        cv2.drawMarker(frame, (cx, cy), GREEN, cv2.MARKER_CROSS, 18, 2)
        tid = f"#{subject.track_id}" if subject.track_id is not None else ""
        label = f"TARGET {tid} {subject.kind}"
        if subject.distance_m is not None:
            label += f"  ~{subject.distance_m:.2f} m"
        cv2.putText(frame, label, (subject.x, subject.y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)


def draw_hud(frame, state, fps):
    h, w = frame.shape[:2]
    lines = [
        f"Lens: {state['lens']}  ({state['lens_points']} pts)",
        f"Focus pos: {state['focus_position']:.2f}",
        f"Distance: " + (f"{state['distance_m']:.2f} m" if state["distance_m"] else "--"),
        f"Status: " + ("LOST (holding focus)" if state["lost"] else "TRACKING"),
    ]
    if state["calibrating"]:
        lines.append("CALIBRATING (auto-focus paused)")
    if config.DRAW_FPS:
        lines.append(f"FPS: {fps:.0f}")

    # Semi-transparent HUD panel.
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (280, 22 * len(lines) + 10), BLACK, -1)
    cv2.addWeighted(panel, 0.45, frame, 0.55, 0, frame)
    for i, text in enumerate(lines):
        color = AMBER if "CALIBRATING" in text else WHITE
        cv2.putText(frame, text, (10, 22 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Focus-ring bar on the right.
    bar_x = w - 30
    bar_color = AMBER if state["calibrating"] else GREEN
    cv2.rectangle(frame, (bar_x, 20), (bar_x + 12, h - 20), WHITE, 1)
    fill_h = int((h - 40) * state["focus_position"])
    cv2.rectangle(frame, (bar_x, h - 20 - fill_h), (bar_x + 12, h - 20), bar_color, -1)

    help_text = (
        "c: calibrate | i/k: rack focus | a: add point | s: save"
        if state["calibrating"] else
        "click: lock focus | l: lens | c: calibrate | s: save | q: quit"
    )
    cv2.putText(frame, help_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)


def main():
    args = parse_args()
    if args.no_mesh:
        config.ENABLE_FACE_MESH = False
    if args.no_pose:
        config.ENABLE_POSE = False

    pipeline = AutofocusPipeline()
    cap = open_capture(args.source)

    # Mouse click -> tap-to-track.
    latest = {"detections": []}

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
            latest["detections"] = state["detections"]

            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            draw_overlay(frame, state)
            draw_hud(frame, state, fps)

            cv2.imshow(config.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l"):
                pipeline.lenses.cycle_active()
            elif key == ord("c"):
                if pipeline.focus.manual_override:
                    pipeline.focus.resume_auto()
                else:
                    pipeline.focus.manual_override = True
            elif key == ord("i") and pipeline.focus.manual_override:
                pipeline.focus.nudge_manual(config.CALIBRATION_STEP)
            elif key == ord("k") and pipeline.focus.manual_override:
                pipeline.focus.nudge_manual(-config.CALIBRATION_STEP)
            elif key == ord("a") and pipeline.focus.manual_override:
                if state["distance_m"] is not None:
                    pipeline.lenses.active_profile().add_point(
                        state["distance_m"], pipeline.focus.position
                    )
            elif key == ord("s"):
                pipeline.save_lens_profiles()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()


if __name__ == "__main__":
    main()
