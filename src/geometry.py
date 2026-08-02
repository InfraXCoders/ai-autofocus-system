"""Small geometry helpers shared across modules (tracking, distance fusion)."""


def iou_xywh(a_xywh, b_xywh) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a_xywh
    bx, by, bw, bh = b_xywh
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
