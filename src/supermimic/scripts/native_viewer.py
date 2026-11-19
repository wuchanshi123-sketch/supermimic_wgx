"""Startup and shutdown helpers for the native MuJoCo player."""

import sys
import time
import warnings
from collections.abc import Callable
from concurrent.futures import Future, wait
from threading import Thread

from mujoco.viewer import Handle


class NativeViewerStartupTimeout(RuntimeError):
  """MuJoCo did not return a viewer handle during startup."""


def setup_native_viewer[T](
  setup: Callable[[], T], close: Callable[[T | None], None]
) -> T:
  if sys.platform == "darwin":
    return setup()

  # launch_passive can wait indefinitely if the window closes during loading.
  ready: Future[T] = Future()

  def launch() -> None:
    try:
      result = setup()
    except BaseException as exc:
      if ready.set_running_or_notify_cancel():
        ready.set_exception(exc)
      else:
        close(None)
    else:
      if ready.set_running_or_notify_cancel():
        ready.set_result(result)
      else:
        close(result)

  Thread(target=launch, name="MuJoCo viewer startup", daemon=True).start()
  try:
    done, _ = wait((ready,), timeout=10.0)
  except KeyboardInterrupt:
    if not ready.cancel():
      close(ready.result() if ready.exception() is None else None)
    raise
  if not done and ready.cancel():
    raise NativeViewerStartupTimeout(
      "MuJoCo viewer did not finish opening. The window may have been closed "
      "during startup."
    )
  return ready.result()


def close_native_viewer(handle: Handle | None) -> None:
  if handle is None:
    return
  if handle.is_running():
    handle.close()

  # Wait for the daemon render thread to release Simulate before the Python
  # GLFW exit handler can run. is_running() only observes the close request.
  deadline = time.monotonic() + 5.0
  while handle._sim() is not None:
    if time.monotonic() >= deadline:
      warnings.warn("MuJoCo renderer did not finish shutting down", stacklevel=2)
      break
    time.sleep(0.01)
