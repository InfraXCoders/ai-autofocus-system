"""Module 5: Lens Database.

Every lens maps "distance in meters" to a physical focus-ring position
differently. This module stores that mapping (the focus curve) per lens, plus
user calibration points.

PHASE GOAL:
  * Store calibration profiles per lens
  * Provide distance_m -> focus_position lookup (interpolated)
  * Save / load user calibrations to disk (JSON)

CURRENT STATE: in-memory profiles with linear interpolation between calibration
points. Persistence is a documented TODO.
"""

from dataclasses import dataclass, field


@dataclass
class LensProfile:
    name: str
    # Calibration points: list of (distance_m, focus_position 0..1)
    # 0.0 = infinity end of the ring, 1.0 = closest-focus end (convention).
    calibration: list[tuple[float, float]] = field(default_factory=list)

    def focus_position(self, distance_m: float) -> float:
        """Interpolate the focus-ring position for a given distance."""
        if not self.calibration:
            # No calibration: assume a naive inverse relationship.
            return max(0.0, min(1.0, 1.0 / max(distance_m, 0.1) * 0.5))

        pts = sorted(self.calibration)
        if distance_m <= pts[0][0]:
            return pts[0][1]
        if distance_m >= pts[-1][0]:
            return pts[-1][1]
        for (d0, p0), (d1, p1) in zip(pts, pts[1:]):
            if d0 <= distance_m <= d1:
                t = (distance_m - d0) / (d1 - d0)
                return p0 + t * (p1 - p0)
        return pts[-1][1]


class LensDatabase:
    def __init__(self):
        self._lenses: dict[str, LensProfile] = {}
        self._seed_defaults()
        self.active: str | None = "Generic 50mm"

    def _seed_defaults(self):
        # A couple of rough starter profiles for demoing the pipeline.
        self.add(LensProfile(
            name="Generic 50mm",
            calibration=[(0.5, 1.0), (1.0, 0.7), (2.0, 0.45), (5.0, 0.2), (30.0, 0.0)],
        ))
        self.add(LensProfile(
            name="Generic 35mm",
            calibration=[(0.4, 1.0), (1.0, 0.6), (3.0, 0.3), (10.0, 0.1), (30.0, 0.0)],
        ))

    def add(self, profile: LensProfile):
        self._lenses[profile.name] = profile

    def get(self, name: str) -> LensProfile | None:
        return self._lenses.get(name)

    def active_profile(self) -> LensProfile:
        return self._lenses[self.active]

    def names(self) -> list[str]:
        return list(self._lenses)

    # TODO(Phase 3): load_from_disk() / save_to_disk() using JSON.
