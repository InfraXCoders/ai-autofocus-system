"""Module 1: AI Vision Engine.

Detects subjects (faces, eyes, bodies) in each camera frame and returns them
in a standard `Detection` format the rest of the pipeline can consume.
"""

from .types import Detection, Point
from .vision_engine import VisionEngine

__all__ = ["VisionEngine", "Detection", "Point"]
