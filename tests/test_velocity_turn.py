import pytest
import torch

from supermimic.utils.velocity_turn import (
  TURN_MEMORY_DIM,
  TurnRewardCfg,
  evaluate_turn,
  turn_command_gate,
  turn_critic_context,
)


def test_turn_shaping_has_smooth_gates_and_is_opt_in():
  cmd = torch.tensor(
    [
      [0, 0, 0],
      [0.5, 0, 0],
      [0.2, 0, 0.5],
      [0, 0, 0.35],
      [0, 0, 0.5],
      [0, 0, -0.5],
      [0, 0, 1.1],
      [0, 0, 1.3],
    ]
  )
  torch.testing.assert_close(
    turn_command_gate(cmd, TurnRewardCfg(enabled=True)),
    torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0]),
  )
  assert not turn_command_gate(cmd, TurnRewardCfg()).any()


_DT = 0.02


def _advance(
  memory,
  contact=(True, True),
  height=(0.0, 0.0),
  *,
  root_xy=(0.0, 0.0),
  yaw=0.5,
  wz_cmd=0.5,
):
  contact = torch.tensor([contact])
  return evaluate_turn(
    memory,
    torch.tensor([[0.0, 0.0, wz_cmd]]),
    torch.tensor([root_xy]),
    torch.tensor([yaw]),
    torch.tensor([height]),
    contact,
    contact.float() * 0.5,
    torch.zeros(1, 2),
    _DT,
    TurnRewardCfg(enabled=True),
  )


def _swing(memory, foot, steps=18, lift=0.05, yaw=0.5):
  for _ in range(steps):
    memory, terms = _advance(
      memory,
      contact=(foot == 1, foot == 0),
      height=(lift, 0.0) if foot == 0 else (0.0, lift),
      yaw=yaw,
    )
    assert terms.alternating.item() == 0.0
  return memory, terms


def test_valid_alternation_is_counted_once_and_same_foot_is_rejected():
  memory, _ = _advance(torch.zeros(1, TURN_MEMORY_DIM))
  memory, terms = _swing(memory, 0)
  assert terms.clearance.item() == pytest.approx(0.25)
  memory, terms = _advance(memory)
  assert terms.alternating.item() == pytest.approx(0.04 / _DT)
  memory, terms = _advance(memory)
  assert terms.alternating.item() == 0.0
  memory, terms = _swing(memory, 0)
  assert terms.clearance.item() == 0.0
  memory, terms = _advance(memory)
  assert terms.alternating.item() == 0.0
  memory, _ = _swing(memory, 1)
  _, terms = _advance(memory)
  assert terms.alternating.item() == pytest.approx(0.04 / _DT)


@pytest.mark.parametrize(
  "steps,lift,yaw",
  [(2, 0.05, 0.5), (40, 0.05, 0.5), (18, 0.002, 0.5), (18, 0.05, -0.5)],
)
def test_landing_requires_enough_clearance_and_correct_turn_progress(steps, lift, yaw):
  memory, _ = _advance(torch.zeros(1, TURN_MEMORY_DIM))
  memory, terms = _swing(memory, 0, steps=steps, lift=lift, yaw=yaw)
  if steps == 40:
    assert terms.clearance.item() == 0.0
  _, terms = _advance(memory, yaw=yaw)
  assert terms.alternating.item() == 0.0


def test_slow_reference_swing_can_complete_in_point_seven_seconds():
  memory, _ = _advance(torch.zeros(1, TURN_MEMORY_DIM))
  memory, terms = _swing(memory, 0, steps=35)
  assert terms.clearance.item() == pytest.approx(0.25)
  _, terms = _advance(memory)
  assert terms.alternating.item() == pytest.approx(0.04 / _DT)


def test_hopping_without_support_gets_no_bonus():
  memory, _ = _advance(torch.zeros(1, TURN_MEMORY_DIM))
  for _ in range(18):
    memory, terms = _advance(memory, contact=(False, False), height=(0.05, 0.05))
    assert terms.clearance.item() == 0.0
  _, terms = _advance(memory)
  assert terms.alternating.item() == 0.0


def test_turn_drift_has_deadband_and_reanchors_on_direction_change():
  memory, _ = _advance(torch.zeros(1, TURN_MEMORY_DIM), root_xy=(3.0, 6.0))
  memory, terms = _advance(memory, root_xy=(3.04, 6.0))
  assert terms.drift_error.item() == 0.0
  memory, terms = _advance(memory, root_xy=(3.12, 6.0))
  assert terms.drift_error.item() == pytest.approx(
    ((0.12 - 0.06) / 0.60) ** 2, abs=1e-7
  )
  torch.testing.assert_close(
    turn_critic_context(memory, torch.tensor([[3.12, 6.0]]))[:, 1:3],
    torch.tensor([[0.12, 0.0]]),
  )
  memory, terms = _advance(memory, root_xy=(3.12, 6.0), wz_cmd=-0.5)
  assert terms.drift_error.item() == 0.0
  memory, _ = _advance(memory, wz_cmd=0.0)
  assert not memory.any()
  assert not turn_critic_context(memory, torch.tensor([[3.12, 6.0]])).any()


def test_turn_clearance_has_gradient_at_small_early_lift():
  height = torch.tensor([[0.001, 0.0]], requires_grad=True)
  _, terms = evaluate_turn(
    torch.zeros(1, TURN_MEMORY_DIM),
    torch.tensor([[0.0, 0.0, 0.5]]),
    torch.zeros(1, 2),
    torch.tensor([0.5]),
    height,
    torch.tensor([[0.8, 1.0]]),
    torch.tensor([[0.5, 0.5]]),
    torch.tensor([[0.5, 1.0]]),
    _DT,
    TurnRewardCfg(enabled=True),
  )
  assert terms.twist.item() == pytest.approx(-0.30 * 0.5 * (0.5**2 + 1.0**2))
  terms.clearance.sum().backward()
  assert height.grad is not None and height.grad[0, 0] > 0.0
