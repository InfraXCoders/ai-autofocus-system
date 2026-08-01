"""Module 3: LiDAR Fusion Engine.

PHASE 3 GOAL: produce ONE reliable distance-to-subject in meters by fusing:
  * LiDAR distance reading (accurate, but a single point / sparse)
  * AI depth estimate      (dense, but relative/noisy)

Fusion matters because LiDAR may hit the background *behind* the subject, while
AI depth drifts. Combining them (e.g. a confidence-weighted average, or using
AI to pick WHICH LiDAR point corresponds to the tracked subject) gives a stable
distance the Focus Engine can trust.

CURRENT STATE: no hardware yet. We estimate distance from apparent face size
(bigger face = closer) using a rough pinhole-camera approximation. This lets
Phase 3/4 be developed before the LiDAR arrives, then swapped in cleanly.
"""

from ..vision.types import Detection

# Rough constants for the face-size distance heuristic.
# distance ~= (real_face_width * focal_px) / face_width_px
_AVG_FACE_WIDTH_M = 0.16       # average human face width ~16cm
_ASSUMED_FOCAL_PX = 650.0      # ballpark for a 720p webcam; calibrate per camera


class DistanceEstimator:
    def __init__(self, min_m: float = 0.3, max_m: float = 30.0):
        self.min_m = min_m
        self.max_m = max_m

    def estimate(self, subject: Detection, lidar_reading_m: float | None = None) -> float:
        """Return best distance in meters for the tracked subject.

        Args:
            subject: the tracked Detection (from the Tracking Engine).
            lidar_reading_m: real LiDAR value once hardware exists (Phase 4+).
        """
        ai_distance = self._distance_from_face_size(subject)

        if lidar_reading_m is None:
            fused = ai_distance
        else:
            # Simple confidence-weighted fusion placeholder.
            # Real version: reject LiDAR if it disagrees wildly (background hit).
            fused = 0.7 * lidar_reading_m + 0.3 * ai_distance

        subject.distance_m = self._clamp(fused)
        return subject.distance_m

    def _distance_from_face_size(self, subject: Detection) -> float:
        # Use face width when available; fall back to body height scaling.
        width_px = subject.w if subject.kind == "face" else max(1, subject.w)
        if width_px <= 0:
            return self.max_m
        d = (_AVG_FACE_WIDTH_M * _ASSUMED_FOCAL_PX) / width_px
        return self._clamp(d)

    def _clamp(self, d: float) -> float:
        return max(self.min_m, min(self.max_m, d))
