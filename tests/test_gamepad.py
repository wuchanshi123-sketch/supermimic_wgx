"""Gamepad event decoding and device lifetime, using simulated device reads."""

from __future__ import annotations

import errno
import struct
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from supermimic.utils import gamepad


def _axis(index: int, value: int, initial: bool = False) -> bytes:
  return struct.pack("=IhBB", 0, value, 0x82 if initial else 0x02, index)


@pytest.fixture
def device(monkeypatch):
  stream = deque()
  opened = Mock(return_value=5)
  closed = Mock()

  def read(_fd, _length):
    if not stream:
      raise BlockingIOError(errno.EAGAIN, "device queue empty")
    data = stream.popleft()
    if isinstance(data, OSError):
      raise data
    return data

  monkeypatch.setattr(
    gamepad,
    "os",
    SimpleNamespace(O_RDONLY=0, O_NONBLOCK=2048, open=opened, close=closed, read=read),
  )
  return stream, opened, closed


def test_ioctl_maps_the_gamepad_axis_codes_before_decoding(device, monkeypatch):
  fcntl = pytest.importorskip("fcntl")
  # Put a trigger ahead of the right stick to distinguish Linux axis *codes*
  # from joystick axis indices.
  codes = bytes((0x01, 0x02, 0x03, 0x00, 0x04))
  requests = {0x80016A11: bytes([len(codes)]), 0x80406A32: codes.ljust(64, b"\x00")}
  monkeypatch.setattr(fcntl, "ioctl", lambda _, op, __: requests[op])
  stream, _, _ = device
  stream.append(_axis(0, -32767, True) + _axis(2, 32767, True) + _axis(3, -32767, True))
  with gamepad.Gamepad("/dev/input/js1") as pad:
    assert pad.read() == (1.0, 1.0, -1.0)
    stream.append(_axis(1, -32767))  # Trigger must not alter the command.
    assert pad.read() == (1.0, 1.0, -1.0)


def test_waits_for_initialized_axes_and_partial_packets_then_applies_deadzone(
  device, monkeypatch
):
  monkeypatch.setattr(gamepad, "_joystick_axes", lambda *_: (0, 1, 3))
  stream, _, _ = device
  with gamepad.Gamepad("/dev/input/js1", deadzone=0.1) as pad:
    stream.append(_axis(1, -32767, True))
    assert pad.read() == (0.0, 0.0, 0.0)
    stream.append(_axis(0, 1000, True) + _axis(3, -16384, True))
    assert pad.read() == pytest.approx((1.0, 0.0, 0.44446), abs=1e-5)
    neutral = _axis(1, 300)
    stream.append(neutral[:4])
    assert pad.read() == (0.0, 0.0, 0.0)
    stream.append(neutral[4:] + _axis(3, -2500))
    assert pad.read() == (0.0, 0.0, 0.0)


def test_disconnect_and_reconnect_clear_stale_axis_values(device, monkeypatch):
  monkeypatch.setattr(gamepad, "_joystick_axes", lambda *_: (0, 1, 3))
  clock = SimpleNamespace(now=0.0)
  monkeypatch.setattr(gamepad, "time", SimpleNamespace(monotonic=lambda: clock.now))
  stream, opened, closed = device
  pad = gamepad.Gamepad("/dev/input/js1")
  stream.append(
    _axis(0, -32767, True) + _axis(1, -32767, True) + _axis(3, -32767, True)
  )
  assert pad.read() == (1.0, 1.0, 1.0)
  stream.append(OSError(errno.ENODEV, "device disconnected"))
  assert pad.read() == (0.0, 0.0, 0.0)
  clock.now = 0.5
  assert pad.read() == (0.0, 0.0, 0.0)
  assert opened.call_count == 1
  clock.now = 1.0
  stream.append(_axis(0, 0, True))
  assert pad.read() == (0.0, 0.0, 0.0)
  stream.append(_axis(1, -32767, True) + _axis(3, 0, True))
  assert pad.read() == (1.0, 0.0, 0.0)
  assert opened.call_count == 2
  pad.close()
  pad.close()
  clock.now = 3.0
  assert pad.read() == (0.0, 0.0, 0.0)
  assert closed.call_count == 2
  assert opened.call_count == 2


def test_mapping_startup_failure_closes_device_and_names_path(device, monkeypatch):
  fcntl = pytest.importorskip("fcntl")
  monkeypatch.setattr(fcntl, "ioctl", lambda *_: b"\x02\x00")
  _, opened, closed = device
  path = Path("/dev/input/js7")
  with pytest.raises(gamepad.GamepadError, match=r"/dev/input/js7.*ABS_RX"):
    gamepad.Gamepad(path)
  opened.assert_called_once_with(path, gamepad.os.O_RDONLY | gamepad.os.O_NONBLOCK)
  closed.assert_called_once_with(5)
