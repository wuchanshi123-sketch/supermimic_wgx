import os
import struct
import sys
from types import SimpleNamespace

import pytest

from supermimic.utils import gamepad

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux joystick ABI")


def _axis_event(axis: int, value: int, *, initial: bool = False) -> bytes:
  return struct.pack("=IhBB", 0, value, 0x82 if initial else 0x02, axis)


def test_gamepad_uses_deadzone_and_stays_neutral_after_disconnect(monkeypatch):
  device_fds = [os.pipe2(os.O_NONBLOCK), os.pipe2(os.O_NONBLOCK)]
  pending_reads = iter(fd for fd, _ in device_fds)
  now = [0.0]

  def connect(self):
    self._fd = os.dup(next(pending_reads))
    self._axes = (0, 1, 3)

  monkeypatch.setattr(gamepad.Gamepad, "_connect", connect)
  monkeypatch.setattr(gamepad, "time", SimpleNamespace(monotonic=lambda: now[0]))

  try:
    with gamepad.Gamepad("/dev/input/js-test") as pad:
      _, writer = device_fds[0]
      packet = _axis_event(0, -32768, initial=True)
      os.write(writer, packet[:4])
      assert pad.read() == (0.0, 0.0, 0.0)
      os.write(writer, packet[4:] + _axis_event(1, 16384, initial=True))
      assert pad.read() == (0.0, 0.0, 0.0)
      os.write(writer, _axis_event(3, 32767, initial=True))
      assert pad.read() == pytest.approx((-0.44446, 1.0, -1.0), abs=1e-5)

      os.write(writer, _axis_event(0, -1000) + _axis_event(5, 32767))
      assert pad.read()[1] == 0.0
      os.close(writer)
      device_fds[0] = (device_fds[0][0], -1)
      assert pad.read() == (0.0, 0.0, 0.0)

      now[0] = 2.0
      assert pad.read() == (0.0, 0.0, 0.0)
      os.write(
        device_fds[1][1],
        b"".join(
          _axis_event(axis, value, initial=True)
          for axis, value in ((0, 0), (1, -32767), (3, -32767))
        ),
      )
      assert pad.read() == (1.0, 0.0, 1.0)
    pad.close()
    assert pad.read() == (0.0, 0.0, 0.0)
  finally:
    for pair in device_fds:
      for fd in pair:
        if fd >= 0:
          os.close(fd)


def test_gamepad_reports_missing_device(tmp_path):
  device = tmp_path / "js-test"
  with pytest.raises(gamepad.GamepadError, match=f"Cannot open joystick {device}"):
    gamepad.Gamepad(device)
