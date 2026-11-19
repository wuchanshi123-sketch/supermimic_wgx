"""Nonblocking Linux joystick input for simulated velocity control."""

from __future__ import annotations

import math
import os
import struct
import sys
import time
from pathlib import Path

# Linux joydev ABI: struct js_event, JSIOCGAXES, JSIOCGAXMAP (linux/joystick.h).
_EVENT = struct.Struct("=IhBB")
_AXIS_EVENT = 0x02
_INIT_EVENT = 0x80
_GET_AXES = 0x80016A11
_GET_AXIS_MAP = 0x80406A32
_AXIS_MAP_SIZE = 64
_STICKS = (0x00, 0x01, 0x03)  # ABS_X, ABS_Y, ABS_RX
_NEUTRAL = (0.0, 0.0, 0.0)


class GamepadError(OSError):
  """The requested joystick could not be opened or configured."""


def _joystick_axes(
  fd: int, overrides: tuple[int, int, int] | None
) -> tuple[int, int, int]:
  import fcntl

  count = fcntl.ioctl(fd, _GET_AXES, b"\x00")[0]
  if overrides is not None:
    if any(axis >= count for axis in overrides):
      raise OSError(f"Axis override is outside the device's {count} joystick axes")
    return overrides

  codes = fcntl.ioctl(fd, _GET_AXIS_MAP, bytes(_AXIS_MAP_SIZE))[:count]
  try:
    left_x, left_y, right_x = (codes.index(code) for code in _STICKS)
  except ValueError as exc:
    raise OSError(
      "Joystick does not report standard left X, left Y and right X axes"
      " (ABS_X, ABS_Y, ABS_RX). Supply the device's three raw axis indices."
    ) from exc
  return left_x, left_y, right_x


class Gamepad:
  """Read normalized ``(vx, vy, wz)`` from two analog sticks.

  ``axes`` optionally overrides the raw (left X, left Y, right X) indices.
  Reads stay neutral until all three axes have reported state. After a
  disconnect, subsequent reads retry opening the same path once per second.
  """

  def __init__(
    self,
    device: Path | str,
    deadzone: float = 0.1,
    *,
    axes: tuple[int, int, int] | None = None,
  ) -> None:
    if not sys.platform.startswith("linux"):
      raise GamepadError("Gamepad input requires Linux and a /dev/input/js* device")
    if not math.isfinite(deadzone) or not 0 <= deadzone < 1:
      raise ValueError("Gamepad deadzone must be in [0, 1)")
    if axes is not None and (len(set(axes)) != 3 or min(axes) < 0):
      raise ValueError(
        "Gamepad axis overrides must be three distinct nonnegative indices"
      )

    self.device = Path(device).expanduser()
    self._deadzone = deadzone
    self._overrides = axes
    self._fd: int | None = None
    self._values: dict[int, int] = {}
    self._partial = b""
    self._initializing = True
    self._next_reconnect = 0.0
    self._closed = False
    self._connect()

  def _connect(self) -> None:
    try:
      fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as exc:
      raise GamepadError(
        f"Cannot open joystick {self.device}: {exc.strerror or exc}"
      ) from exc
    try:
      self._axes = _joystick_axes(fd, self._overrides)
    except OSError as exc:
      os.close(fd)
      raise GamepadError(
        f"Cannot read joystick axis mapping for {self.device}: {exc}"
      ) from exc
    self._fd = fd

  def _disconnect(self) -> None:
    if self._fd is not None:
      os.close(self._fd)
      self._fd = None
    self._values.clear()
    self._partial = b""
    self._initializing = True
    self._next_reconnect = time.monotonic() + 1.0

  def _normalize(self, value: int) -> float:
    magnitude = min(abs(value) / 32767.0, 1.0)
    return math.copysign(
      max(0.0, magnitude - self._deadzone) / (1.0 - self._deadzone), value
    )

  def read(self) -> tuple[float, float, float]:
    """Return the current axes without waiting for a new event."""
    if self._closed:
      return _NEUTRAL
    if self._fd is None:
      if time.monotonic() < self._next_reconnect:
        return _NEUTRAL
      try:
        self._connect()
      except OSError:
        self._next_reconnect = time.monotonic() + 1.0
        return _NEUTRAL

    assert self._fd is not None
    # Drain the queue in batches while bounding time spent on a busy device.
    for _ in range(8):
      try:
        data = os.read(self._fd, _EVENT.size * 64)
      except BlockingIOError:
        break
      except OSError:
        data = b""
      if not data:
        self._disconnect()
        return _NEUTRAL

      packet = self._partial + data
      length = len(packet) - len(packet) % _EVENT.size
      self._partial = packet[length:]
      for _, value, kind, axis in _EVENT.iter_unpack(packet[:length]):
        init = bool(kind & _INIT_EVENT)
        if init and not self._initializing:
          # Driver queue overflow starts a fresh batch of initial axis values.
          self._values.clear()
        self._initializing = init
        if (kind & ~_INIT_EVENT) == _AXIS_EVENT and axis in self._axes:
          self._values[axis] = value
    else:
      return _NEUTRAL  # The device queue has not drained yet.

    if self._partial or len(self._values) != len(self._axes):
      return _NEUTRAL
    left_x, left_y, right_x = (self._values[axis] for axis in self._axes)
    return (
      self._normalize(-left_y),  # Up is forward.
      self._normalize(-left_x),  # Left is positive lateral velocity.
      self._normalize(-right_x),  # Left is positive yaw rate.
    )

  def close(self) -> None:
    self._closed = True
    self._disconnect()

  def __enter__(self) -> Gamepad:
    return self

  def __exit__(self, *_exc) -> None:
    self.close()
