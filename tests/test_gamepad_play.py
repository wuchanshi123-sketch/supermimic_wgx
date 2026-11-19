from types import SimpleNamespace
from unittest.mock import Mock

import torch

from supermimic.scripts.viewers import SuperMimicNativeViewer, SuperMimicViewerMixin


def test_native_step_polls_once_and_sends_the_scaled_velocity_goal(monkeypatch):
  viewer = object.__new__(SuperMimicNativeViewer)
  viewer._gamepad = Mock()
  viewer._gamepad.read.side_effect = [(0.5, -0.25, -1.0), (0.0, 0.0, 0.0)]
  viewer._gamepad_ranges = ((-1.5, 2.0), (-0.8, 0.5), (-2.0, 3.0))
  viewer._gamepad_motion = SimpleNamespace(velocity_goal=None)
  viewer._raw_goal_state = {"velocity": torch.zeros(1, 3)}
  viewer._is_paused = True
  observed = []

  def step(self, *, force=False):
    if self._is_paused and not force:
      return False
    goal = self._raw_goal_state["velocity"]
    assert self._gamepad_motion.velocity_goal is goal
    observed.append(goal.clone())
    return True

  monkeypatch.setattr(SuperMimicViewerMixin, "step_and_predict", step)
  assert not viewer.step_and_predict()
  viewer._gamepad.read.assert_not_called()
  assert viewer.step_and_predict(force=True)
  viewer._gamepad.read.assert_called_once_with()
  torch.testing.assert_close(observed[-1], torch.tensor([[1.0, -0.2, -2.0]]))
  viewer._is_paused = False
  assert viewer.step_and_predict()
  assert viewer._gamepad.read.call_count == 2
  torch.testing.assert_close(observed[-1], torch.zeros(1, 3))
