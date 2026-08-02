"""Module 5: Lens Database.

Every lens maps "distance in meters" to a physical focus-ring position
differently. This module stores that mapping (the focus curve) per lens, plus
user calibration points.

PHASE GOAL:
  * Store calibration profiles per lens
  * Provide distance_m -> focus_position lookup (interpolated)
  * Save / load user calibrations to disk (JSON)

CURRENT STATE (Phase 3): in-memory profiles with linear interpolation between
calibration points, plus JSON persistence so a user's calibration survives
across sessions.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


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

    def add_point(self, distance_m: float, position: float) -> None:
        """Record a user calibration point (rack focus at a known distance).

        Replaces any existing point at (nearly) the same distance so
        re-calibrating a point doesn't create duplicates.
        """
        position = max(0.0, min(1.0, position))
        for i, (d, _) in enumerate(self.calibration):
            if abs(d - distance_m) < 0.05:
                self.calibration[i] = (distance_m, position)
                return
        self.calibration.append((distance_m, position))


class LensDatabase:
    def __init__(self, profiles_path: str | Path | None = None):
        self._lenses: dict[str, LensProfile] = {}
        self.active: str | None = None
        self.profiles_path = Path(profiles_path) if profiles_path else None

        if not (self.profiles_path and self.load_from_disk(self.profiles_path)):
            self._seed_defaults()
            self.active = "Generic 50mm"

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

    def add_lens(self, name: str) -> LensProfile:
        """Create a fresh, uncalibrated lens profile and make it active."""
        profile = LensProfile(name=name, calibration=[])
        self.add(profile)
        self.active = name
        return profile

    def get(self, name: str) -> LensProfile | None:
        return self._lenses.get(name)

    def active_profile(self) -> LensProfile:
        return self._lenses[self.active]

    def names(self) -> list[str]:
        return list(self._lenses)

    def cycle_active(self) -> str:
        """Switch to the next lens in the list (wraps around). Returns its name."""
        names = self.names()
        idx = names.index(self.active)
        self.active = names[(idx + 1) % len(names)]
        return self.active

    # ---- Persistence ----
    def save_to_disk(self, path: str | Path | None = None) -> None:
        path = Path(path) if path else self.profiles_path
        if path is None:
            raise ValueError("No path given and no default profiles_path configured")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active": self.active,
            "lenses": [
                {"name": p.name, "calibration": [list(pt) for pt in p.calibration]}
                for p in self._lenses.values()
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    def load_from_disk(self, path: str | Path) -> bool:
        """Load profiles from JSON. Returns True on success, False if the file
        doesn't exist yet (not an error — first run seeds defaults instead)."""
        path = Path(path)
        if not path.exists():
            return False

        data = json.loads(path.read_text())
        lenses = data.get("lenses", [])
        if not lenses:
            return False

        self._lenses = {}
        for entry in lenses:
            self.add(LensProfile(
                name=entry["name"],
                calibration=[tuple(pt) for pt in entry.get("calibration", [])],
            ))
        active = data.get("active")
        self.active = active if active in self._lenses else next(iter(self._lenses))
        return True
