"""Module 2: Tracking Engine.

PHASE 2 GOAL: lock onto ONE subject and follow it frame-to-frame, even when
detection flickers. Responsibilities:
  * Subject Lock          — operator taps a subject; we commit to it
  * Motion Prediction     — estimate where it will be next frame (Kalman filter)
  * Multi-frame Tracking  — bridge gaps when detection drops for a few frames
  * Re-identification     — recognize the same subject after it reappears

CURRENT STATE: minimal "nearest-to-last-position" tracker so the pipeline runs
end-to-end today. Replace the internals in Phase 2 (e.g. with a Kalman filter
+ appearance embeddings) without changing this interface.
"""

from ..vision.types import Detection


class SubjectTracker:
    def __init__(self, lost_grace_frames: int = 15):
        self.lost_grace_frames = lost_grace_frames
        self._locked_center: tuple[int, int] | None = None
        self._frames_since_seen = 0
        self._next_id = 1
        self.active_id: int | None = None

    def lock_on(self, detection: Detection) -> None:
        """Operator selected a subject (tap-to-track)."""
        self._locked_center = detection.center
        self.active_id = self._next_id
        self._next_id += 1
        self._frames_since_seen = 0

    def update(self, detections: list[Detection]) -> Detection | None:
        """Return the currently tracked subject, or None if lost.

        Placeholder logic: pick the face/body nearest the last known position.
        Auto-locks the largest face if nothing is locked yet.
        """
        candidates = [d for d in detections if d.kind in ("face", "body")]
        if not candidates:
            self._frames_since_seen += 1
            return None

        if self._locked_center is None:
            # Auto-lock the largest face (or body) as a sensible default.
            target = max(candidates, key=lambda d: d.area)
            self.lock_on(target)
            target.track_id = self.active_id
            return target

        # Nearest to last known center.
        cx, cy = self._locked_center

        def dist2(d: Detection) -> int:
            dx, dy = d.center[0] - cx, d.center[1] - cy
            return dx * dx + dy * dy

        target = min(candidates, key=dist2)
        self._locked_center = target.center
        self._frames_since_seen = 0
        target.track_id = self.active_id
        return target

    def is_lost(self) -> bool:
        return self._frames_since_seen > self.lost_grace_frames
