"""Native MuJoCo viewer startup and shutdown shared by both playback modes."""

import sys
import time
import warnings
from concurrent.futures import Future, wait
from threading import Thread

from mjlab.viewer import NativeMujocoViewer


class NativeViewerStartupTimeout(RuntimeError):
  """MuJoCo did not return a viewer handle during startup."""


class ManagedNativeMujocoViewer(NativeMujocoViewer):
  """Wait for the renderer to release its window before the process exits."""

  def setup(self) -> None:
    if sys.platform == "darwin":
      super().setup()
      return

    # Closing the GLFW window while MuJoCo is loading can leave launch_passive
    # waiting for a handle that will never arrive. Keep this wait off the main
    # thread so the CLI can exit if startup stalls.
    setup = super().setup
    ready: Future[None] = Future()

    def launch() -> None:
      error = None
      try:
        setup()
      except BaseException as exc:
        error = exc
      if not ready.set_running_or_notify_cancel():
        self.close()  # Clean up a handle that arrived after startup was abandoned.
      elif error is None:
        ready.set_result(None)
      else:
        ready.set_exception(error)

    Thread(target=launch, name="MuJoCo viewer startup", daemon=True).start()
    try:
      done, _ = wait((ready,), timeout=10.0)
    except KeyboardInterrupt:
      if not ready.cancel():
        self.close()
      raise
    if not done and ready.cancel():
      raise NativeViewerStartupTimeout(
        "MuJoCo viewer did not finish opening. The window may have been closed "
        "during startup."
      )
    ready.result()

  def close(self) -> None:
    handle, self.viewer = self.viewer, None
    if handle is None:
      return
    if handle.is_running():
      handle.close()

    # MuJoCo reports the close request before the daemon render thread tears
    # down its window. Wait for its Simulate object to be released before
    # Python's GLFW exit handler can run.
    deadline = time.monotonic() + 5.0
    while handle._sim() is not None:
      if time.monotonic() >= deadline:
        warnings.warn("MuJoCo renderer did not finish shutting down", stacklevel=2)
        break
      time.sleep(0.01)
