"""Module 6: Wireless Communication.

Carries focus commands from the Vision Pod to the lens focus motor with low
latency and reliable delivery.

PHASE GOAL:
  * Low-latency link (target < 20ms) — ESP-NOW / BLE / Nordic
  * Reliable packet handling (sequence numbers, ack, drop stale commands)
  * OTA firmware updates

CURRENT STATE: a "virtual motor" that just prints the command. Swap this for a
pyserial/USB or BLE transport in Phase 4 without touching the pipeline.
"""


class MotorLink:
    def __init__(self, transport: str = "virtual"):
        self.transport = transport
        self._seq = 0
        self._last_sent: float | None = None

    def send_focus(self, position: float) -> None:
        """Send a focus-ring position (0..1) to the motor.

        Only sends when the value meaningfully changed, to avoid flooding the
        link (a form of rate limiting the real transport will also want).
        """
        if self._last_sent is not None and abs(position - self._last_sent) < 0.002:
            return
        self._seq += 1
        self._last_sent = position
        if self.transport == "virtual":
            # Real version: pack {seq, position} and write to serial/BLE.
            # Kept silent by default; enable for debugging.
            pass

    @property
    def last_position(self) -> float | None:
        return self._last_sent
