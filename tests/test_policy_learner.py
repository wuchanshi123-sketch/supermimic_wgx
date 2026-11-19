import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
  gae_returns,
)
from supermimic.models import TaskCritics

NUM_ENVS = 2


class _DummyActor(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(()))

  def forward(self, obs: TensorDict) -> torch.Tensor:
    return obs["actor"] * self.weight


class _DummyRepresentation(nn.Module):
  dt = 1.0


class _TwoStepEnv:
  """Reward 1 per step; the first step terminates with probability 0.5."""

  num_envs = NUM_ENVS
  num_actions = 1
  max_step_reward = float("inf")

  def __init__(self):
    self.dynamics = nn.Linear(1, 1)
    self.representation = _DummyRepresentation()
    self.step_idx = 0

  def sample_initial_states(self) -> TensorDict:
    return TensorDict({}, batch_size=[NUM_ENVS])

  def reset(self, batch: TensorDict) -> TensorDict:
    del batch
    self.step_idx = 0
    return self._obs()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    rewards = 1.0 + actions.squeeze(-1)
    term_prob = torch.full((NUM_ENVS,), 0.5 if self.step_idx == 0 else 0.0)
    self.step_idx += 1
    return self._obs(), rewards, term_prob, {}

  def _obs(self) -> TensorDict:
    obs = {"actor": torch.zeros(NUM_ENVS, 1), "critic": torch.zeros(NUM_ENVS, 1)}
    return TensorDict(obs, batch_size=[NUM_ENVS])


class _ConstantCritic(nn.Module):
  def __init__(self, value: float):
    super().__init__()
    self.value = nn.Parameter(torch.tensor(value))

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return obs * 0.0 + self.value


def test_survival_masking_counts_current_transition_reward_before_done():
  learner = BpttPolicyLearner(
    _DummyActor(),
    _TwoStepEnv(),
    BpttPolicyLearnerCfg(horizon=2, gamma=1.0, survival_masking=True),
  )

  stats = learner.update()

  # Return 1 + (1 - 0.5) * 1, averaged over the horizon.
  assert stats["Policy/reward"].item() == pytest.approx(1.5 / 2)


def test_actor_critic_bootstraps_the_surviving_last_state():
  learner = ActorCriticPolicyLearner(
    _DummyActor(),
    _ConstantCritic(1.0),
    _TwoStepEnv(),
    ActorCriticPolicyLearnerCfg(horizon=2, gamma=0.99, survival_masking=True),
  )

  stats = learner.update()

  # Return 1 + 0.99 * 0.5 * 1, plus the value 1 discounted by 0.99^2 and by
  # survival 0.5, averaged over the horizon.
  assert stats["Policy/reward"].item() == pytest.approx((1.495 + 0.49005) / 2)
  assert torch.isfinite(stats["Policy/value_function"])


def test_gae_returns_discount_by_soft_done():
  returns = gae_returns(
    values=torch.zeros(2, 1),
    last_value=torch.zeros(1),
    rewards=torch.ones(2, 1),
    dones=torch.tensor([[0.5], [0.0]]),
    gamma=0.99,
    lam=1.0,
  )

  assert torch.allclose(returns, torch.tensor([[1.495], [1.0]]))


def test_task_critics_select_their_own_inputs_in_shuffled_rows():
  motion = nn.Linear(4, 1, bias=False)
  velocity = nn.Linear(5, 1, bias=False)
  with torch.no_grad():
    motion.weight.copy_(torch.tensor([[1.0, 10.0, 100.0, 1000.0]]))
    velocity.weight.copy_(torch.tensor([[1.0, 10.0, 100.0, 1000.0, 10000.0]]))
  critics = TaskCritics(motion, velocity, rep_state_dim=1, motion_goal_dim=2)
  # [state, motion x2, context, command x3, is_velocity]
  obs = torch.tensor([[2.0, 3, 4, 5, 6, 7, 8, 1], [2.0, 3, 4, 5, 6, 7, 8, 0]])
  torch.testing.assert_close(critics(obs), torch.tensor([[58762.0], [5432.0]]))


def test_task_critics_bootstrap_with_each_tasks_reward_ceiling():
  wm = _TwoStepEnv()
  wm.max_step_reward = 1.0
  wm.velocity_max_step_reward = 2.0
  motion = nn.Linear(3, 1)
  velocity = nn.Linear(5, 1)
  with torch.no_grad():
    motion.bias.fill_(10.0)
    velocity.bias.fill_(10.0)
  learner = ActorCriticPolicyLearner(
    _DummyActor(),
    TaskCritics(motion, velocity, rep_state_dim=1, motion_goal_dim=1),
    wm,
    ActorCriticPolicyLearnerCfg(horizon=2, gamma=0.5),
  )
  critic_obs = torch.zeros(NUM_ENVS, 7)
  critic_obs[1, -1] = 1
  obs = TensorDict({"critic": critic_obs}, batch_size=[NUM_ENVS])
  result = learner._bootstrap(obs, torch.ones(NUM_ENVS), torch.zeros(NUM_ENVS))
  torch.testing.assert_close(result, torch.tensor([0.5, 1.0]))
