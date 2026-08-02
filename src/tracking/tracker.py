"""Module 2: Tracking Engine (Phase 2 — real multi-object tracking).

A SORT-style tracker built on:
  * Kalman filter per subject   -> motion prediction + smoothing   (kalman.py)
  * IoU data association        -> match detections to tracks
  * Appearance re-identification -> reconnect a subject after a gap  (reid.py)
  * Track lifecycle             -> tentative / confirmed / lost / deleted (track.py)

Responsibilities from the spec, and how they map here:
  * Subject Lock         -> `select_at()` pins `active_id` to one track
  * Motion Prediction    -> Kalman `predict()` every frame
  * Multi-frame Tracking -> tracks survive `max_age` frames without detection
  * Re-identification    -> unmatched detections matched to lost tracks by appearance

Interface is unchanged for the pipeline: `update()`, `select_at()`, `is_lost()`,
`active_id` — only the internals got real.
"""

from ..geometry import iou_xywh
from ..vision.types import Detection
from . import reid
from .track import Track


class SubjectTracker:
    def __init__(
        self,
        lost_grace_frames: int = 15,
        iou_threshold: float = 0.3,
        reid_threshold: float = 0.6,
        n_init: int = 2,
        max_age: int = 30,
    ):
        self.lost_grace_frames = lost_grace_frames
        self.iou_threshold = iou_threshold
        self.reid_threshold = reid_threshold
        self.n_init = n_init
        self.max_age = max_age

        self.tracks: list[Track] = []
        self._next_id = 1
        self.active_id: int | None = None

    # ---- Public API (used by the pipeline) ----
    def update(self, detections: list[Detection], frame_bgr) -> Detection | None:
        """Advance all tracks with this frame's detections; return focus target."""
        # Track faces preferentially; fall back to bodies when no face is visible.
        faces = [d for d in detections if d.kind == "face"]
        dets = faces if faces else [d for d in detections if d.kind == "body"]

        # 1) Predict every existing track forward.
        for t in self.tracks:
            t.predict()

        # 2) Appearance descriptor for each detection (for re-ID).
        descs = [reid.compute_descriptor(frame_bgr, (d.x, d.y, d.w, d.h)) for d in dets]

        # 3) Associate by IoU (greedy, highest overlap first).
        matches, unmatched_dets, unmatched_tracks = self._match_by_iou(dets)
        for di, ti in matches:
            d = dets[di]
            self.tracks[ti].update((d.x, d.y, d.w, d.h), descs[di], d.confidence)

        # 4) Re-identify: try to reconnect leftover detections to lost tracks
        #    by appearance (they won't overlap, so IoU can't find them).
        still_unmatched = []
        for di in unmatched_dets:
            best_ti, best_sim = None, self.reid_threshold
            for ti in unmatched_tracks:
                sim = reid.similarity(descs[di], self.tracks[ti].descriptor)
                if sim > best_sim:
                    best_sim, best_ti = sim, ti
            if best_ti is not None:
                d = dets[di]
                self.tracks[best_ti].update((d.x, d.y, d.w, d.h), descs[di], d.confidence)
                unmatched_tracks.remove(best_ti)
            else:
                still_unmatched.append(di)

        # 5) Spawn new tracks for genuinely new detections.
        for di in still_unmatched:
            d = dets[di]
            self.tracks.append(Track(
                self._next_id, (d.x, d.y, d.w, d.h), descs[di], d.kind,
                confidence=d.confidence, n_init=self.n_init, max_age=self.max_age,
            ))
            self._next_id += 1

        # 6) Retire tracks unseen for too long.
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return self._focus_target()

    def select_at(self, px: int, py: int) -> None:
        """Tap-to-track: lock focus onto the track under this pixel."""
        hits = [t for t in self.tracks if t.contains(px, py)]
        if hits:
            # Smallest containing box = most specific pick.
            self.active_id = min(hits, key=lambda t: t.area).id

    def is_lost(self) -> bool:
        active = self._active_track()
        if active is None:
            return True
        return active.time_since_update > self.lost_grace_frames

    # ---- Internals ----
    def _match_by_iou(self, dets):
        matches, pairs = [], []
        for di, d in enumerate(dets):
            for ti, t in enumerate(self.tracks):
                iou = iou_xywh((d.x, d.y, d.w, d.h), t.bbox_xywh)
                if iou >= self.iou_threshold:
                    pairs.append((iou, di, ti))

        pairs.sort(reverse=True)  # highest IoU first
        used_d, used_t = set(), set()
        for _, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            matches.append((di, ti))
            used_d.add(di)
            used_t.add(ti)

        unmatched_dets = [i for i in range(len(dets)) if i not in used_d]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_t]
        return matches, unmatched_dets, unmatched_tracks

    def _active_track(self) -> Track | None:
        if self.active_id is not None:
            for t in self.tracks:
                if t.id == self.active_id:
                    return t
        return None

    def _focus_target(self) -> Detection | None:
        active = self._active_track()

        # Auto-select if nothing locked (or the locked track was deleted):
        # the largest confirmed track (closest subject).
        if active is None:
            pool = [t for t in self.tracks if t.confirmed] or self.tracks
            if not pool:
                self.active_id = None
                return None
            active = max(pool, key=lambda t: t.area)
            self.active_id = active.id

        # Beyond the grace window we stop reporting a subject so the Focus
        # Engine HOLDS (safety) rather than chasing a stale prediction.
        if active.time_since_update > self.lost_grace_frames:
            return None

        x, y, w, h = active.bbox_xywh
        return Detection(
            kind=active.kind,
            confidence=active.confidence,
            x=x, y=y, w=w, h=h,
            track_id=active.id,
        )
