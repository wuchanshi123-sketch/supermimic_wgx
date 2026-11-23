"""Exercise the R00 model through the real standalone velocity environment."""

import pytest
import torch

from supermimic.scripts.train_utils import create_env
from supermimic.tasks.r00_velocity_env_cfg import r00_velocity_env_cfg


def test_r00_first_physics_step_exports_full_state_and_reward():
  env = create_env(r00_velocity_env_cfg(), num_envs=2, device="cpu")
  try:
    assert env.unwrapped.step_dt == pytest.approx(0.02)
    assert env.num_actions == 31
    assert env.unwrapped.scene["robot"].body_names[0] == "base_link"
    obs, rewards, dones, _ = env.step(torch.zeros((2, 31)))
    assert obs["full_state"].shape == (2, 75)
    assert obs["actor"].shape == (2, 99)
    assert torch.isfinite(obs["full_state"]).all()
    assert torch.isfinite(rewards).all()
    assert not dones.any()
  finally:
    env.close()
