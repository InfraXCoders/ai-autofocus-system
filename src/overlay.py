"""Overlay drawing shared by every front-end (desktop OpenCV window and the
web demo). Keeping this here means the desktop app (main.py) and the web app
(web/server.py) draw identical HUDs from the same pipeline `state` dict.

  * green box  = tracked subject (the focus target)
  * cyan boxes = other detected faces
  * yellow     = eyes
  * magenta    = body/person
  * HUD        = distance estimate, focus-ring position, lens, FPS
"""

import cv2

from .geometry import iou_xywh

# Colors (BGR)
GREEN = (0, 220, 0)
CYAN = (220, 220, 0)
YELLOW = (0, 220, 220)
MAGENTA = (220, 0, 220)
WHITE = (240, 240, 240)
BLACK = (0, 0, 0)
AMBER = (0, 170, 255)


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


def draw_hud(frame, state, fps, draw_fps=True, help_text=None):
    h, w = frame.shape[:2]
    lines = [
        f"Lens: {state['lens']}  ({state['lens_points']} pts)",
        f"Focus pos: {state['focus_position']:.2f}",
        f"Distance: " + (f"{state['distance_m']:.2f} m" if state["distance_m"] else "--"),
        f"Status: " + ("LOST (holding focus)" if state["lost"] else "TRACKING"),
    ]
    if state["calibrating"]:
        lines.append("CALIBRATING (auto-focus paused)")
    if draw_fps:
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

    if help_text is None:
        help_text = (
            "c: calibrate | i/k: rack focus | a: add point | s: save"
            if state["calibrating"] else
            "click: lock focus | l: lens | c: calibrate | s: save | q: quit"
        )
    cv2.putText(frame, help_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)


def handle_key(pipeline, key: str, last_distance_m: float | None, calibration_step: float):
    """Apply one keyboard/button command to the pipeline. Shared by the
    desktop key loop and the web app's button/keyboard messages."""
    if key == "l":
        pipeline.lenses.cycle_active()
    elif key == "c":
        if pipeline.focus.manual_override:
            pipeline.focus.resume_auto()
        else:
            pipeline.focus.manual_override = True
    elif key == "i" and pipeline.focus.manual_override:
        pipeline.focus.nudge_manual(calibration_step)
    elif key == "k" and pipeline.focus.manual_override:
        pipeline.focus.nudge_manual(-calibration_step)
    elif key == "a" and pipeline.focus.manual_override and last_distance_m is not None:
        pipeline.lenses.active_profile().add_point(last_distance_m, pipeline.focus.position)
    elif key == "s":
        pipeline.save_lens_profiles()
