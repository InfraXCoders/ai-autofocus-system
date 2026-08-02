"""The autofocus pipeline — wires all 6 backend modules into one flow.

    frame -> VisionEngine -> SubjectTracker -> DistanceEstimator
          -> FocusEngine (+ LensDatabase) -> MotorLink

Phase 1 runs Vision + Tracking for real; the rest run in "simulation" so the
whole loop is demonstrable today and each module can be upgraded independently.
"""

import config
from .vision import VisionEngine
from .tracking import SubjectTracker
from .lidar_fusion import DistanceEstimator
from .focus import FocusEngine
from .lens_db import LensDatabase
from .comms import MotorLink


class AutofocusPipeline:
    def __init__(self):
        self.vision = VisionEngine(
            face_confidence=config.FACE_DETECTION_CONFIDENCE,
            model_selection=config.FACE_MODEL_SELECTION,
            enable_face_mesh=config.ENABLE_FACE_MESH,
            enable_pose=config.ENABLE_POSE,
        )
        self.tracker = SubjectTracker(config.TRACK_LOST_GRACE_FRAMES)
        self.distance = DistanceEstimator(
            config.MIN_FOCUS_DISTANCE_M, config.MAX_FOCUS_DISTANCE_M,
            config.DISTANCE_SMOOTHING,
        )
        self.lenses = LensDatabase(profiles_path=config.LENS_PROFILES_PATH)
        self.focus = FocusEngine(config.FOCUS_SMOOTHING)
        self.motor = MotorLink(transport="virtual")

    def step(self, frame_bgr) -> dict:
        """Process one frame. Returns a state dict for the UI/overlay."""
        detections = self.vision.process(frame_bgr)
        subject = self.tracker.update(detections, frame_bgr)

        distance_m = None
        focus_pos = self.focus.position
        if subject is not None:
            distance_m = self.distance.estimate(subject, detections)  # lidar=None for now
            focus_pos = self.focus.update(distance_m, self.lenses.active_profile())
            self.motor.send_focus(focus_pos)

        return {
            "detections": detections,
            "subject": subject,
            "distance_m": distance_m,
            "focus_position": focus_pos,
            "lens": self.lenses.active,
            "lens_points": len(self.lenses.active_profile().calibration),
            "lost": self.tracker.is_lost(),
            "calibrating": self.focus.manual_override,
        }

    def select_at(self, x: int, y: int) -> None:
        """Tap-to-track: lock the track under the (x, y) click."""
        self.tracker.select_at(x, y)

    def save_lens_profiles(self) -> None:
        self.lenses.save_to_disk()

    def close(self):
        self.vision.close()
