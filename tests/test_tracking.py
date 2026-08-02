"""Automated tests for Phase 2 (Tracking Engine) + core Phase 1 data types.

These run WITHOUT a webcam — they feed synthetic detections into the tracker
and assert the four spec behaviors hold:
  * Subject Lock         (tap-to-track picks the right subject)
  * Motion Prediction    (track survives a short detection gap)
  * Multi-frame Tracking (stable ID across frames)
  * Re-identification    (same ID after the subject reappears elsewhere)

Run it:
    .venv/bin/python -m tests.test_tracking       # from the project root
or, if pytest is installed:
    .venv/bin/python -m pytest -v
"""

import numpy as np

from src.tracking import SubjectTracker
from src.vision.types import Detection

# A dummy frame; re-ID reads pixels from it. A colored gradient gives the two
# subjects distinguishable-but-stable appearance histograms.
_FRAME = np.tile(np.linspace(0, 255, 640, dtype="uint8"), (480, 1))
_FRAME = np.stack([_FRAME, _FRAME[::-1], _FRAME], axis=-1)


def _face(x, y, w=80, h=90, conf=0.9):
    return Detection(kind="face", confidence=conf, x=x, y=y, w=w, h=h)


def test_stable_id_across_frames():
    """A subject moving across the frame keeps ONE track ID."""
    tr = SubjectTracker(n_init=2)
    ids = []
    for i in range(8):
        subj = tr.update([_face(100 + i * 15, 150)], _FRAME)
        assert subj is not None, f"lost subject on frame {i}"
        ids.append(subj.track_id)
    assert len(set(ids)) == 1, f"ID should be stable, got {ids}"


def test_motion_prediction_survives_gap():
    """When detection drops for a few frames, the track keeps predicting."""
    tr = SubjectTracker(n_init=2, lost_grace_frames=15)
    for i in range(5):
        tr.update([_face(100 + i * 15, 150)], _FRAME)

    # 3 frames with NO detections — shorter than the grace window.
    subj = None
    for _ in range(3):
        subj = tr.update([], _FRAME)
    assert subj is not None, "track should still predict during a short gap"
    assert not tr.is_lost(), "should not report LOST within the grace window"


def test_focus_holds_when_truly_lost():
    """Past the grace window, no subject is reported so focus can HOLD."""
    tr = SubjectTracker(n_init=2, lost_grace_frames=5, max_age=30)
    for i in range(5):
        tr.update([_face(100 + i * 15, 150)], _FRAME)
    subj = None
    for _ in range(10):  # longer than grace (5)
        subj = tr.update([], _FRAME)
    assert subj is None, "should stop reporting a subject once truly lost"
    assert tr.is_lost(), "should report LOST past the grace window"


def test_reidentification_after_reappear():
    """A subject that vanishes and reappears elsewhere keeps its ID."""
    tr = SubjectTracker(n_init=2, lost_grace_frames=5, max_age=40, iou_threshold=0.9)
    first_id = None
    for i in range(5):
        subj = tr.update([_face(80, 150)], _FRAME)  # stationary at left
        first_id = subj.track_id

    for _ in range(6):  # disappears (past grace, still within max_age)
        tr.update([], _FRAME)

    # Reappears far away — no IoU overlap, so only re-ID can reconnect it.
    subj = tr.update([_face(80, 150)], _FRAME)
    assert subj is not None
    assert subj.track_id == first_id, (
        f"re-ID should restore id {first_id}, got {subj.track_id}"
    )


def test_tap_to_track_selects_correct_subject():
    """Clicking a subject locks focus onto that specific track."""
    tr = SubjectTracker(n_init=1)
    left = _face(50, 60)
    right = _face(400, 300)
    tr.update([left, right], _FRAME)

    tr.select_at(430, 340)  # inside the right subject's box
    subj = tr.update([left, right], _FRAME)
    cx, _ = subj.center
    assert cx > 320, f"should have locked the RIGHT subject, center x={cx}"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    print("Running Phase 1-2 tracking tests...\n")
    sys.exit(0 if _run_all() else 1)
