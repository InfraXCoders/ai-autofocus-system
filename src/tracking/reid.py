"""Appearance descriptors for subject re-identification.

When a tracked subject disappears (turns away, is occluded, walks out of frame)
and later reappears, IoU matching alone can't reconnect them — the new detection
has no overlap with the old, predicted box. Re-ID solves this by comparing what
the subject *looks like*.

We use a simple, fast HSV color histogram as the appearance descriptor and
compare descriptors by correlation. It's not as strong as a deep embedding, but
it's dependency-free and good enough to reconnect the same person across short
gaps. In a later phase this can be swapped for an ONNX Re-ID embedding model
without changing the tracker.
"""

import cv2
import numpy as np

_H_BINS = 32
_S_BINS = 32


def compute_descriptor(frame_bgr, bbox_xywh):
    """Return a normalized HSV histogram for the crop, or None if invalid."""
    x, y, w, h = bbox_xywh
    H, W = frame_bgr.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return None

    crop = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [_H_BINS, _S_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0.0, 1.0, cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)


def similarity(a, b) -> float:
    """Correlation between two descriptors, clamped to 0..1 (1 = identical)."""
    if a is None or b is None:
        return 0.0
    am = a - a.mean()
    bm = b - b.mean()
    denom = np.linalg.norm(am) * np.linalg.norm(bm)
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(am, bm) / denom, 0.0, 1.0))
