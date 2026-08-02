"""Automated tests for Phase 3 (Distance Fusion, Lens Database, Focus Engine
calibration mode). No webcam needed — everything runs on synthetic data.

Run it:
    .venv/bin/python -m tests.test_focus_and_lens
"""

import tempfile
from pathlib import Path

from src.focus import FocusEngine
from src.lens_db import LensDatabase, LensProfile
from src.lidar_fusion import DistanceEstimator
from src.vision.types import Detection


def _face(x, y, w=80, h=90, track_id=1):
    return Detection(kind="face", confidence=0.9, x=x, y=y, w=w, h=h, track_id=track_id)


def _body(x, y, w=200, h=400, track_id=1):
    return Detection(kind="body", confidence=0.9, x=x, y=y, w=w, h=h, track_id=track_id)


# ---- DistanceEstimator ----

def test_face_only_distance_matches_pinhole_formula():
    """With no secondary cue, distance should equal the single-cue formula."""
    est = DistanceEstimator(smoothing=0.0)  # no smoothing for a direct check
    subject = _face(100, 100, w=162)  # width chosen so distance ~= 0.16*650/162 ~= 0.642m
    d = est.estimate(subject, detections=[subject])
    expected = (0.16 * 650.0) / 162
    assert abs(d - expected) < 0.01, f"expected ~{expected:.3f}, got {d:.3f}"


def test_fusion_blends_face_and_body_cues():
    """When both a face and an overlapping body are visible, the fused
    distance should land between the two single-cue estimates (not equal
    to either extreme), since both cues contribute."""
    est = DistanceEstimator(smoothing=0.0)
    face = _face(150, 60, w=80, h=90, track_id=1)     # a "close" cue
    body = _body(100, 60, w=100, h=800, track_id=1)   # a "far" cue (tall box -> close; short -> far)
    # Make the body cue clearly different from the face-only estimate.
    body = _body(100, 60, w=100, h=200, track_id=1)   # short box -> reads as far away

    face_only = est._cue_for(face)[0]
    body_only = est._cue_for(body)[0]
    assert face_only != body_only, "test setup: cues must differ to prove fusion happened"

    fused = est.estimate(face, detections=[face, body])
    lo, hi = sorted([face_only, body_only])
    assert lo <= fused <= hi, f"fused distance {fused:.3f} should be between {lo:.3f} and {hi:.3f}"


def test_temporal_smoothing_reduces_jitter():
    """A track's distance should move gradually toward a new noisy reading,
    not jump instantly (reliable distance estimation, not raw jitter)."""
    est = DistanceEstimator(smoothing=0.8)  # heavy smoothing
    steady = _face(100, 100, w=160, track_id=7)
    d1 = est.estimate(steady, detections=[steady])

    # Sudden noisy reading (much closer).
    noisy = _face(100, 100, w=320, track_id=7)
    d2 = est.estimate(noisy, detections=[noisy])

    raw_target = est._cue_for(noisy)[0]
    assert abs(d2 - d1) < abs(raw_target - d1), (
        "smoothed distance should move less than the raw jump"
    )


def test_distance_is_clamped_to_bounds():
    est = DistanceEstimator(min_m=0.5, max_m=10.0, smoothing=0.0)
    huge_face = _face(0, 0, w=5000, track_id=1)   # would compute a tiny/negative distance
    tiny_face = _face(0, 0, w=1, track_id=2)      # would compute a huge distance
    assert est.estimate(huge_face, [huge_face]) >= 0.5
    assert est.estimate(tiny_face, [tiny_face]) <= 10.0


# ---- LensProfile / LensDatabase ----

def test_lens_profile_add_point_upserts_near_duplicates():
    lens = LensProfile(name="Test Lens", calibration=[(1.0, 0.5)])
    lens.add_point(1.02, 0.6)  # within 0.05m -> should replace, not append
    assert len(lens.calibration) == 1
    assert lens.calibration[0] == (1.02, 0.6)

    lens.add_point(3.0, 0.2)  # far enough -> should append
    assert len(lens.calibration) == 2


def test_lens_profile_interpolation():
    lens = LensProfile(name="Test Lens", calibration=[(1.0, 1.0), (2.0, 0.0)])
    assert lens.focus_position(1.5) == 0.5  # midpoint
    assert lens.focus_position(0.5) == 1.0  # clamps below range
    assert lens.focus_position(5.0) == 0.0  # clamps above range


def test_lens_database_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lenses.json"

        db1 = LensDatabase(profiles_path=path)  # no file yet -> seeds defaults
        db1.active_profile().add_point(1.23, 0.42)
        db1.save_to_disk()

        db2 = LensDatabase(profiles_path=path)  # loads what db1 saved
        assert db2.active == db1.active
        assert (1.23, 0.42) in db2.active_profile().calibration


def test_lens_database_cycle_active():
    db = LensDatabase()  # in-memory defaults, no disk
    first = db.active
    second = db.cycle_active()
    assert second != first
    assert second in db.names()
    # Cycling through all lenses returns to the start.
    for _ in range(len(db.names()) - 1):
        db.cycle_active()
    assert db.active == first


# ---- FocusEngine calibration mode ----

def test_manual_override_pauses_auto_focus():
    engine = FocusEngine(smoothing=0.0)
    lens = LensProfile(name="L", calibration=[(1.0, 1.0), (10.0, 0.0)])

    engine.update(1.0, lens)  # auto-focus sets position near 1.0
    engine.nudge_manual(-0.3)  # operator racks focus by hand
    pos_after_nudge = engine.position

    # While overridden, update() must NOT move the position even with a
    # very different distance reading.
    held = engine.update(10.0, lens)
    assert held == pos_after_nudge, "auto-focus must not run while calibrating"

    engine.resume_auto()
    resumed = engine.update(10.0, lens)
    assert resumed != pos_after_nudge, "auto-focus should resume after resume_auto()"


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
    print("Running Phase 3 tests (distance fusion, lens DB, calibration)...\n")
    sys.exit(0 if _run_all() else 1)
