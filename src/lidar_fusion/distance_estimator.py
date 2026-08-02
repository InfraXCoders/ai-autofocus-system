"""Module 3: LiDAR Fusion Engine.

PHASE GOAL: produce ONE reliable distance-to-subject in meters by fusing:
  * LiDAR distance reading (accurate, but a single point / sparse)
  * AI depth estimate      (dense, but relative/noisy)

Fusion matters because LiDAR may hit the background *behind* the subject, while
AI depth drifts. Combining them (e.g. a confidence-weighted average, or using
AI to pick WHICH LiDAR point corresponds to the tracked subject) gives a stable
distance the Focus Engine can trust.

PHASE 3 STATE: no LiDAR hardware yet, so "AI depth estimate" is itself a fusion
of two cheap, dependency-free visual cues (a real depth-network is documented
as the upgrade path, see `_distance_from_face` docstring):

  * face width   — reliable, but only available when the face is visible
  * body height  — available more often (works side-on, from a distance),
                    but noisier (clothing/pose change apparent height)

The two cues are combined by a fixed confidence weighting, then the *result*
is exponential-smoothed per track_id so a jittery detection box doesn't cause
the estimated distance (and therefore focus) to jitter frame to frame — this
is the "Reliable Distance Estimation" requirement.
"""

from ..geometry import iou_xywh
from ..vision.types import Detection

# Rough constants for the size-based distance heuristics.
# distance ~= (real_size_m * focal_px) / size_px
_AVG_FACE_WIDTH_M = 0.16        # average human face width ~16cm
_AVG_BODY_HEIGHT_M = 1.65       # approximate framed body height (head-to-hip typical framing)
_ASSUMED_FOCAL_PX = 650.0       # ballpark for a 720p webcam; calibrate per camera

_FACE_CUE_WEIGHT = 0.75         # face width is the more reliable cue
_BODY_CUE_WEIGHT = 0.25


class DistanceEstimator:
    def __init__(self, min_m: float = 0.3, max_m: float = 30.0, smoothing: float = 0.4):
        self.min_m = min_m
        self.max_m = max_m
        self.smoothing = smoothing  # 0 = raw/instant, close to 1 = heavily smoothed
        self._smoothed_by_track: dict[int, float] = {}

    def estimate(
        self,
        subject: Detection,
        detections: list[Detection] | None = None,
        lidar_reading_m: float | None = None,
    ) -> float:
        """Return best distance in meters for the tracked subject.

        Args:
            subject: the tracked Detection (from the Tracking Engine).
            detections: all detections this frame, used to find a secondary
                cue (e.g. the body box behind a tracked face) to fuse with.
            lidar_reading_m: real LiDAR value once hardware exists (Phase 4+).
        """
        ai_distance = self._fuse_visual_cues(subject, detections or [])

        if lidar_reading_m is None:
            fused = ai_distance
        else:
            # Simple confidence-weighted fusion placeholder.
            # Real version: reject LiDAR if it disagrees wildly (background hit).
            fused = 0.7 * lidar_reading_m + 0.3 * ai_distance

        fused = self._clamp(fused)
        smoothed = self._smooth(subject.track_id, fused)
        subject.distance_m = smoothed
        return smoothed

    def _fuse_visual_cues(self, subject: Detection, detections: list[Detection]) -> float:
        cues: list[tuple[float, float]] = []  # (distance_m, weight)

        primary = self._cue_for(subject)
        if primary is not None:
            cues.append(primary)

        secondary_kind = "body" if subject.kind == "face" else "face"
        secondary_det = self._find_overlapping(subject, detections, secondary_kind)
        if secondary_det is not None:
            secondary = self._cue_for(secondary_det)
            if secondary is not None:
                cues.append(secondary)

        if not cues:
            return self.max_m

        total_weight = sum(w for _, w in cues)
        return sum(d * w for d, w in cues) / total_weight

    def _cue_for(self, det: Detection) -> tuple[float, float] | None:
        """Return (distance_m, weight) for one detection, or None if unusable."""
        if det.kind == "face" and det.w > 0:
            d = (_AVG_FACE_WIDTH_M * _ASSUMED_FOCAL_PX) / det.w
            return self._clamp(d), _FACE_CUE_WEIGHT
        if det.kind == "body" and det.h > 0:
            d = (_AVG_BODY_HEIGHT_M * _ASSUMED_FOCAL_PX) / det.h
            return self._clamp(d), _BODY_CUE_WEIGHT
        return None

    def _find_overlapping(self, subject: Detection, detections: list[Detection], kind: str):
        best, best_iou = None, 0.05  # ignore near-zero overlap
        subj_box = (subject.x, subject.y, subject.w, subject.h)
        for d in detections:
            if d.kind != kind:
                continue
            score = iou_xywh(subj_box, (d.x, d.y, d.w, d.h))
            if score > best_iou:
                best_iou, best = score, d
        return best

    def _smooth(self, track_id: int | None, value: float) -> float:
        if track_id is None:
            return value
        prev = self._smoothed_by_track.get(track_id)
        if prev is None:
            smoothed = value
        else:
            a = 1.0 - self.smoothing
            smoothed = prev + a * (value - prev)
        self._smoothed_by_track[track_id] = smoothed
        return smoothed

    def _clamp(self, d: float) -> float:
        return max(self.min_m, min(self.max_m, d))
