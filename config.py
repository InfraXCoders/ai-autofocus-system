"""Central configuration for the AI Autofocus System.

Keeping settings in one place makes it easy to tune the pipeline without
digging through module code.
"""

# ---- Vision Engine ----
FACE_DETECTION_CONFIDENCE = 0.5      # 0..1 — minimum confidence to report a face
FACE_MODEL_SELECTION = 1             # 0 = short range (<2m), 1 = full range (<5m)
ENABLE_FACE_MESH = True              # detailed eye landmarks (slower, more precise)
ENABLE_POSE = True                   # full-body / person detection

# ---- Tracking Engine (Phase 2) ----
TRACK_LOST_GRACE_FRAMES = 15         # keep predicting this many frames after losing subject

# ---- Focus Engine (Phase 4) ----
FOCUS_SMOOTHING = 0.35               # 0 = instant/jittery, 1 = very slow/cinematic
MIN_FOCUS_DISTANCE_M = 0.3           # closest focusable distance (meters)
MAX_FOCUS_DISTANCE_M = 30.0

# ---- Display ----
WINDOW_NAME = "AI Autofocus - Vision Pod (Phase 1)"
DRAW_FPS = True
DRAW_LANDMARKS = True
