"""The autofocus pipeline — wires all 6 backend modules into one flow.

    frame -> VisionEngine -> SubjectTracker -> DistanceEstimator
          -> FocusEngine (+ LensDatabase) -> MotorLink

Phase 1 runs Vision + Tracking for real; the rest run in "simulation" so the
whole loop is demonstrable today and each module can be upgraded independently.
"""

import config
from .vision import VisionEngine, Detection
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
            config.MIN_FOCUS_DISTANCE_M, config.MAX_FOCUS_DISTANCE_M
        )
        self.lenses = LensDatabase()
        self.focus = FocusEngine(config.FOCUS_SMOOTHING)
        self.motor = MotorLink(transport="virtual")

    def step(self, frame_bgr) -> dict:
        """Process one frame. Returns a state dict for the UI/overlay."""
        detections = self.vision.process(frame_bgr)
        subject = self.tracker.update(detections)

        distance_m = None
        focus_pos = self.focus.position
        if subject is not None:
            distance_m = self.distance.estimate(subject)  # lidar=None for now
            focus_pos = self.focus.update(distance_m, self.lenses.active_profile())
            self.motor.send_focus(focus_pos)

        return {
            "detections": detections,
            "subject": subject,
            "distance_m": distance_m,
            "focus_position": focus_pos,
            "lens": self.lenses.active,
            "lost": self.tracker.is_lost(),
        }

    def select_at(self, x: int, y: int, detections: list[Detection]) -> None:
        """Tap-to-track: lock the detection under the (x, y) click."""
        faces = [d for d in detections if d.kind in ("face", "body")]
        for d in faces:
            if d.x <= x <= d.x + d.w and d.y <= y <= d.y + d.h:
                self.tracker.lock_on(d)
                return

    def close(self):
        self.vision.close()
