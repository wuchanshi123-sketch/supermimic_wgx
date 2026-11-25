"""Exercise the R00 model through the real standalone velocity environment."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch

from supermimic.runners.velocity_runner import (
  R00_VELOCITY_AMP_TASK_ID,
  VelocityRunner,
  VelocityRunnerCfg,
)
from supermimic.scripts.train_utils import create_env
from supermimic.sim_context import SimContext
from supermimic.tasks.r00_velocity_env_cfg import r00_velocity_env_cfg
from supermimic.tasks.velocity_env_cfg import configure_turn_reward, velocity_env_cfg
from supermimic.utils.velocity_turn import TURN_MEMORY_DIM, TurnRewardCfg


def test_r00_cylinder_soles_fit_and_mirror():
  cfg = r00_velocity_env_cfg(amp=True)
  model = cfg.scene.entities["robot"].spec_fn().compile()
  names = cfg.events["foot_friction"].params["asset_cfg"].geom_names
  left, right = np.array([model.geom(name).id for name in names]).reshape(2, 7)
  both = np.concatenate((left, right))
  assert np.all(model.geom_type[both] == int(mujoco.mjtGeom.mjGEOM_CYLINDER))
  assert np.all(model.geom_priority[both] == 1)
  assert np.all(model.geom_contype[both] != 0)
  np.testing.assert_allclose(model.geom_pos[left], model.geom_pos[right] * [1, -1, 1])
  np.testing.assert_allclose(model.geom_size[left], model.geom_size[right])
  np.testing.assert_allclose(model.geom_pos[both, 2] - model.geom_size[both, 0], -0.068)
  # Check the cylinders are horizontal along the foot's fore-aft axis.
  axis = np.empty(3)
  for geom in both:
    mujoco.mju_rotVecQuat(axis, np.array([0.0, 0.0, 1.0]), model.geom_quat[geom])
    np.testing.assert_allclose(axis, [1, 0, 0], atol=1e-12)


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


def test_r00_amp_damping_is_per_environment_and_resampled_on_partial_reset():
  cfg = r00_velocity_env_cfg(amp=True)
  cfg.seed = 42
  env = create_env(cfg, num_envs=4, device="cpu")
  try:
    env.reset()
    sim = env.unwrapped.sim
    joints = env.unwrapped.scene["robot"].indexing.joint_v_adr
    values = sim.model.dof_damping[:, joints].clone()
    assert values.shape == (4, 31)
    torch.testing.assert_close(values, values[:, :1].expand_as(values))
    assert torch.all((values >= 0.0) & (values <= 0.3))
    assert torch.unique(values[:, 0]).numel() == 4
    assert torch.count_nonzero(sim.model.dof_damping[:, :6]) == 0

    context = SimContext.from_env(env.unwrapped)
    column = context.find("joint_damping")
    assert column is not None
    torch.testing.assert_close(
      context.read(env.unwrapped)[:, column], (values[:, :1] - 0.15) / 0.15
    )
    env.unwrapped.reset(env_ids=torch.tensor([1]))
    updated = sim.model.dof_damping[:, joints]
    torch.testing.assert_close(updated[[0, 2, 3]], values[[0, 2, 3]])
    assert updated[1, 0] != values[1, 0]
  finally:
    env.close()


def test_passive_damping_randomization_is_scoped_to_r00_amp_training():
  assert "joint_damping" not in r00_velocity_env_cfg().events
  assert "joint_damping" not in velocity_env_cfg(amp=True).events
  assert r00_velocity_env_cfg(play=True, amp=True).events["joint_damping"].params[
    "ranges"
  ] == (0.0, 0.0)


def test_pure_turn_memory_reaches_world_model_and_resets_per_environment():
  cfg = r00_velocity_env_cfg(amp=True)
  cfg.commands["twist"] = replace(
    cfg.commands["twist"],
    lin_vel_x=(0.0, 0.0),
    lin_vel_y=(0.0, 0.0),
    ang_vel_z=(0.8, 0.8),
    standing_fraction=0.0,
    single_axis_fraction=0.0,
  )
  turn_cfg = TurnRewardCfg(enabled=True)
  configure_turn_reward(cfg, turn_cfg)
  env = create_env(cfg, num_envs=2, device="cpu")
  try:
    runner = VelocityRunner(
      env,
      VelocityRunnerCfg(
        buffer_size=8,
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        dynamics_hidden_dims=(8,),
        policy=replace(VelocityRunnerCfg().policy, batch_size=4),
        turn_reward=turn_cfg,
      ),
      R00_VELOCITY_AMP_TASK_ID,
      device="cpu",
    )
    reward = runner._sim_reward
    assert reward.turn_memory is not None
    assert reward.turn_memory.shape == (2, TURN_MEMORY_DIM)
    assert torch.count_nonzero(reward.turn_memory) == 0
    runner.step()
    assert torch.all(reward.turn_metrics["gate"] == 1.0)

    sample = runner.wm_env.sample_initial_states()
    # First transition stores the pre-step snapshot, not the updated turn anchor.
    assert torch.count_nonzero(sample["turn_memory"]) == 0
    observations = runner.wm_env.reset(sample)
    assert (
      observations["critic"].shape[-1] == runner.policy_learner.critic[0].in_features
    )
    assert observations["actor"].shape[-1] == 102
    actions = torch.zeros((4, 31), requires_grad=True)
    _, imagined_reward, _, _ = runner.wm_env.step(actions)
    assert torch.isfinite(imagined_reward).all()
    (grad,) = torch.autograd.grad(imagined_reward.sum(), actions)
    assert torch.isfinite(grad).all()

    saved = reward.turn_memory[1].clone()
    assert torch.count_nonzero(saved) > 0
    with torch.inference_mode():
      env.unwrapped.reset(env_ids=torch.tensor([0]))
    assert torch.count_nonzero(reward.turn_memory[0]) == 0
    torch.testing.assert_close(reward.turn_memory[1], saved)
  finally:
    env.close()
