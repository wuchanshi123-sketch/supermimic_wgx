from types import SimpleNamespace

import torch

from supermimic.utils.velocity_collision import (
  SELF_COLLISION_HISTORY,
  VelocityRewardSignalModel,
  self_collision_count,
  self_collision_sensor_cfg,
)


def test_self_collision_count_matches_unitree_force_history():
  cfg = self_collision_sensor_cfg()
  assert cfg.history_length == SELF_COLLISION_HISTORY == 4
  assert cfg.num_slots == 1 and cfg.reduce == "none"
  assert cfg.primary == cfg.secondary
  assert (cfg.primary.entity, cfg.primary.mode, cfg.primary.pattern) == (
    "robot",
    "subtree",
    "pelvis",
  )
  history = torch.zeros(2, 2, 4, 3)
  history[0, 0, 0, 1] = 10.0  # Equality does not count.
  history[0, 0, 1, 0] = 11.0
  history[0, 1, 1, 2] = 12.0  # Two contacts in the same substep count once.
  history[0, 1, 3, 1] = -11.0
  history[1, :, :, 1] = 11.0
  env = SimpleNamespace(
    scene={cfg.name: SimpleNamespace(data=SimpleNamespace(force_history=history))}
  )
  torch.testing.assert_close(self_collision_count(env), torch.tensor([2.0, 4.0]))


def test_reward_signal_model_fits_batches_and_provides_action_gradients():
  torch.manual_seed(24)
  model = VelocityRewardSignalModel(6, 3, 2)
  states = torch.randn(2, 3, 6)
  actions = torch.randn(2, 3, 3, requires_grad=True)
  contexts = torch.randn(2, 3, 2)
  targets = torch.tensor(
    [
      [[0.0, 0.0, 0.0, 0.0], [4e5, 1.0, 0.25, 150.0], [5e5, 2.0, 0.5, 0.0]],
      [[6e5, 4.0, 1.0, 250.0], [2e4, 0.0, 0.0, 0.0], [4e5, 3.0, 0.25, 50.0]],
    ]
  )
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  predictions = model(states, actions, contexts)
  start_loss = model.loss(predictions, targets)
  start_loss.backward()
  optimizer.step()
  predictions = model(states, actions, contexts)
  assert predictions.shape == targets.shape
  assert torch.isfinite(predictions).all()
  assert torch.all(predictions[..., 0] >= 0)
  assert torch.all((0 <= predictions[..., 1]) & (predictions[..., 1] <= 4))
  assert torch.all(predictions[..., 2:] >= 0)
  assert model.loss(predictions, targets) < start_loss.detach()
  for column in (2, 3):
    (action_grad,) = torch.autograd.grad(
      predictions[..., column].sum(), (actions,), retain_graph=column == 2
    )
    assert torch.isfinite(action_grad).all() and action_grad.abs().sum() > 0


def test_reward_signal_checkpoint_loads_legacy_outputs_without_changing_foot_rows():
  source = VelocityRewardSignalModel(6, 3, 2)
  with torch.no_grad():
    source.network[-1].weight[:2].fill_(0.25)
    source.network[-1].bias[:2].copy_(torch.tensor([1.0, -1.0]))
  key = f"network.{len(source.network) - 1}"
  legacy = source.state_dict()
  legacy[f"{key}.weight"] = legacy[f"{key}.weight"][:2].clone()
  legacy[f"{key}.bias"] = legacy[f"{key}.bias"][:2].clone()

  target = VelocityRewardSignalModel(6, 3, 2)
  initial_foot_weight = target.network[-1].weight[2:].detach().clone()
  initial_foot_bias = target.network[-1].bias[2:].detach().clone()
  target.load_checkpoint(legacy)

  inputs = torch.zeros(2, 6), torch.zeros(2, 3), torch.zeros(2, 2)
  torch.testing.assert_close(target(*inputs)[..., :2], source(*inputs)[..., :2])
  torch.testing.assert_close(target.network[-1].weight[2:], initial_foot_weight)
  torch.testing.assert_close(target.network[-1].bias[2:], initial_foot_bias)
  assert legacy[f"{key}.bias"].shape == (2,)
