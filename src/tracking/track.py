"""A single tracked subject: Kalman motion state + appearance + lifecycle.

Lifecycle:
  tentative  -> just created, needs `n_init` hits before we trust it
  confirmed  -> matched enough times; eligible to be the focus target
  lost       -> not matched this frame (time_since_update > 0), still predicting
  deleted    -> unmatched for longer than `max_age`; dropped
"""

from .kalman import KalmanBoxTracker


def xywh_to_cxcywh(bbox):
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0, w, h


def cxcywh_to_xywh(cxcywh):
    cx, cy, w, h = cxcywh
    return int(cx - w / 2.0), int(cy - h / 2.0), int(w), int(h)


class Track:
    def __init__(self, track_id, bbox_xywh, descriptor, kind,
                 confidence=1.0, n_init=2, max_age=30):
        self.id = track_id
        self.kind = kind
        self.confidence = confidence
        self.kf = KalmanBoxTracker(xywh_to_cxcywh(bbox_xywh))
        self.descriptor = descriptor

        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.n_init = n_init
        self.max_age = max_age
        self.confirmed = n_init <= 1

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_xywh, descriptor, confidence=None):
        self.kf.update(xywh_to_cxcywh(bbox_xywh))
        self.hits += 1
        self.time_since_update = 0
        if confidence is not None:
            self.confidence = confidence
        if descriptor is not None:
            if self.descriptor is None:
                self.descriptor = descriptor
            else:
                # Blend appearance slowly so it adapts but resists flicker.
                self.descriptor = 0.9 * self.descriptor + 0.1 * descriptor
        if self.hits >= self.n_init:
            self.confirmed = True

    @property
    def bbox_xywh(self):
        x, y, w, h = cxcywh_to_xywh(self.kf.cxcywh)
        return max(0, x), max(0, y), max(1, w), max(1, h)

    @property
    def area(self):
        _, _, w, h = self.bbox_xywh
        return w * h

    def contains(self, px, py) -> bool:
        x, y, w, h = self.bbox_xywh
        return x <= px <= x + w and y <= py <= y + h

    def is_deleted(self) -> bool:
        return self.time_since_update > self.max_age
