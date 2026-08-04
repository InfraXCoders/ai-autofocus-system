"""Automated tests for Phase 4 (Wireless Communication / MotorLink).

No ESP32 hardware needed — a fake serial port stands in for pyserial so we
can verify the packet protocol (framing, sequencing, rate limiting, acks)
without a real device attached. This is the same protocol the ESP32 firmware
(firmware/esp32_focus_motor/esp32_focus_motor.ino) parses on the wire.

Run it:
    .venv/bin/python -m tests.test_comms
"""

from src.comms import MotorLink


class FakeSerial:
    """Records writes and lets a test queue up fake incoming lines, standing
    in for pyserial.Serial without any real hardware."""

    def __init__(self, port, baud, timeout=0):
        self.port = port
        self.baud = baud
        self.written = []       # list of raw bytes written
        self._incoming = b""    # bytes queued for the next readline()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def queue_incoming(self, line: str) -> None:
        self._incoming += (line + "\n").encode("ascii")

    @property
    def in_waiting(self) -> int:
        return len(self._incoming)

    def readline(self) -> bytes:
        if b"\n" not in self._incoming:
            return b""
        line, _, rest = self._incoming.partition(b"\n")
        self._incoming = rest
        return line + b"\n"

    def close(self) -> None:
        self.closed = True


def _make_link():
    fake = FakeSerial("fake-port", 115200)
    link = MotorLink(transport="serial", port="fake-port",
                      serial_factory=lambda *a, **k: fake)
    return link, fake


def test_virtual_transport_never_writes_to_serial():
    link = MotorLink(transport="virtual")
    link.send_focus(0.5)
    link.send_focus(0.9)
    assert link.last_position == 0.9  # state still tracked, just no hardware I/O


def test_serial_packet_format():
    link, fake = _make_link()
    link.send_focus(0.673)
    assert len(fake.written) == 1
    assert fake.written[0] == b"F1 0.673\n"


def test_sequence_number_increments_per_send():
    link, fake = _make_link()
    link.send_focus(0.1)
    link.send_focus(0.5)   # big enough change to not be rate-limited
    link.send_focus(0.9)
    seqs = [int(pkt.split(b" ")[0][1:]) for pkt in fake.written]
    assert seqs == [1, 2, 3]


def test_tiny_changes_are_rate_limited():
    """Sub-threshold position changes shouldn't flood the link."""
    link, fake = _make_link()
    link.send_focus(0.500)
    link.send_focus(0.501)  # < 0.002 delta -> should be suppressed
    link.send_focus(0.700)  # big change -> should send
    assert len(fake.written) == 2


def test_poll_ack_parses_esp32_response():
    link, fake = _make_link()
    fake.queue_incoming("A7")
    assert link.poll_ack() == 7


def test_poll_ack_returns_none_when_nothing_queued():
    link, fake = _make_link()
    assert link.poll_ack() is None


def test_poll_ack_ignores_malformed_lines():
    link, fake = _make_link()
    fake.queue_incoming("garbage")
    assert link.poll_ack() is None


def test_serial_transport_requires_port():
    try:
        MotorLink(transport="serial", serial_factory=lambda *a, **k: FakeSerial("x", 9600))
        assert False, "expected ValueError for missing port"
    except ValueError:
        pass


def test_close_closes_the_underlying_serial_port():
    link, fake = _make_link()
    link.close()
    assert fake.closed is True


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    print("Running Phase 4 tests (motor link serial protocol, no hardware needed)...\n")
    sys.exit(0 if _run_all() else 1)
