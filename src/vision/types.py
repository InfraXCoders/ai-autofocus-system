"""Shared data types for the vision pipeline.

Every module speaks in these types so modules stay decoupled — the Tracking
Engine, LiDAR Fusion, and Focus Engine all consume `Detection` objects without
caring how they were produced.
"""

from dataclasses import dataclass, field


@dataclass
class Point:
    """A normalized point (0..1) relative to the frame, plus pixel coords."""

    x: float          # normalized 0..1
    y: float          # normalized 0..1
    px: int = 0       # pixel x
    py: int = 0       # pixel y


@dataclass
class Detection:
    """A single detected subject in a frame.

    This is the universal currency of the pipeline. The focus target is
    ultimately chosen from a list of these.
    """

    kind: str                       # "face" | "eye" | "body" | "object"
    confidence: float               # 0..1
    # Bounding box in pixel coordinates
    x: int
    y: int
    w: int
    h: int
    # Optional keypoints (e.g. eyes on a face)
    keypoints: dict = field(default_factory=dict)   # name -> Point
    # Filled in by later modules:
    track_id: int | None = None     # set by Tracking Engine (Phase 2)
    distance_m: float | None = None  # set by LiDAR Fusion (Phase 3)

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def area(self) -> int:
        return self.w * self.h
