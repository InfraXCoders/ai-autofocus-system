"""Module 6: Wireless Communication.

Carries focus commands from the Vision Pod to the lens focus motor with low
latency and reliable delivery.

PHASE GOAL:
  * Low-latency link (target < 20ms) — ESP-NOW / BLE / Nordic
  * Reliable packet handling (sequence numbers, ack, drop stale commands)
  * OTA firmware updates

PHASE 4 STATE (pre-hardware): a real `transport="serial"` link to an ESP32
over USB, using a small line protocol the firmware understands:

    Mac -> ESP32:  "F<seq> <position>\\n"   e.g. "F42 0.673\\n"
    ESP32 -> Mac:  "A<seq>\\n"              e.g. "A42\\n"        (acknowledgement)

Sequence numbers let the firmware silently drop a stale/out-of-order packet
instead of racking focus backwards — the "reliable packet handling"
requirement. See firmware/esp32_focus_motor/esp32_focus_motor.ino for the
receiving side, and docs/hardware.md for wiring + setup.

This has been tested against the packet-framing logic only (tests/test_comms.py,
via an injected fake serial port) — actual hardware is on order, not yet in
hand. `transport="virtual"` remains the default so the rest of the pipeline
keeps working with no hardware attached at all.

Real ESP-NOW/BLE wireless (instead of a USB cable) is a documented next step
once this serial link is validated against real hardware.
"""

try:
    import serial as _pyserial  # pyserial; optional until hardware is connected
except ImportError:
    _pyserial = None


class MotorLink:
    def __init__(
        self,
        transport: str = "virtual",
        port: str | None = None,
        baud: int = 115200,
        serial_factory=None,
    ):
        """
        Args:
            transport: "virtual" (no hardware, default) or "serial" (ESP32 over USB).
            port: serial device path, e.g. "/dev/tty.usbserial-0001" (macOS) or
                "COM5" (Windows). Required when transport="serial".
            baud: must match Serial.begin(...) in the ESP32 firmware.
            serial_factory: advanced/testing — a callable(port, baud, timeout=0)
                used instead of pyserial.Serial. Lets tests inject a fake port
                without real hardware.
        """
        self.transport = transport
        self._seq = 0
        self._last_sent: float | None = None
        self._serial = None

        if transport == "serial":
            factory = serial_factory
            if factory is None:
                if _pyserial is None:
                    raise RuntimeError(
                        "transport='serial' requires pyserial: pip install pyserial"
                    )
                factory = _pyserial.Serial
            if not port:
                raise ValueError(
                    "transport='serial' requires a port, e.g. '/dev/tty.usbserial-0001'"
                )
            self._serial = factory(port, baud, timeout=0)

    def send_focus(self, position: float) -> None:
        """Send a focus-ring position (0..1) to the motor.

        Only sends when the value meaningfully changed, to avoid flooding the
        link (a form of rate limiting the real transport will also want).
        """
        if self._last_sent is not None and abs(position - self._last_sent) < 0.002:
            return
        self._seq += 1
        self._last_sent = position

        if self._serial is not None:
            packet = f"F{self._seq} {position:.3f}\n".encode("ascii")
            self._serial.write(packet)
        # "virtual": no-op, used for development without hardware attached.

    def poll_ack(self) -> int | None:
        """Non-blocking read of the ESP32's most recently acknowledged
        sequence number, or None if nothing new has arrived. Useful for a
        link-health indicator (e.g. "motor connected" in the UI)."""
        if self._serial is None:
            return None
        waiting = getattr(self._serial, "in_waiting", 0)
        if not waiting:
            return None
        line = self._serial.readline().decode("ascii", errors="ignore").strip()
        if not line.startswith("A"):
            return None
        try:
            return int(line[1:])
        except ValueError:
            return None

    @property
    def last_position(self) -> float | None:
        return self._last_sent

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
