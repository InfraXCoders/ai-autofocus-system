"""Module 4: Focus Engine — the heart of the product.

Takes the fused distance-to-subject and the active lens's focus curve, and
produces a smooth focus-ring target for the motor.

PHASE GOAL:
  * distance_m + lens curve -> focus position
  * Smooth "cinematic" transitions (no jitter, no hunting)
  * Respect a latency budget (focus must feel responsive, < ~100ms)

CURRENT STATE: exponential smoothing between the previous and target positions.
This is the safety-critical module: if the subject is lost it should HOLD the
last focus, never hunt wildly.
"""

from ..lens_db import LensProfile


class FocusEngine:
    def __init__(self, smoothing: float = 0.35):
        # smoothing: 0 = instant (jittery), 1 = very slow (cinematic drift)
        self.smoothing = smoothing
        self._current_position: float = 0.0  # 0..1 focus-ring position
        self._has_target = False

    def update(self, distance_m: float | None, lens: LensProfile) -> float:
        """Return the smoothed focus-ring position (0..1) to send to the motor.

        If distance is None (subject lost), HOLD the current position — a
        deliberate safety choice so focus never hunts when tracking drops.
        """
        if distance_m is None:
            return self._current_position

        target = lens.focus_position(distance_m)

        if not self._has_target:
            # First lock — snap to target so we don't drift in from 0.
            self._current_position = target
            self._has_target = True
        else:
            a = 1.0 - self.smoothing
            self._current_position += a * (target - self._current_position)

        return self._current_position

    @property
    def position(self) -> float:
        return self._current_position
