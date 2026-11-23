"""Exercise reference state initialization through the real R00 reset path."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
)

from supermimic.scripts.train_utils import create_env
from supermimic.tasks.r00_velocity_env_cfg import get_r00_spec, r00_velocity_env_cfg
from supermimic.tasks.velocity_rsi import (
  REFERENCE_STATE_RESET_TERM,
  reference_state_reset_event,
)
from supermimic.utils.gencoord_helpers import decompose_state


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
  model = get_r00_spec().compile()
  hinges = [
    j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
  ]
  names = tuple(model.joint(j).name for j in reversed(hinges))
  joint_pos = model.jnt_range[hinges[::-1]].mean(axis=-1).astype(np.float32)
  joint_vel = np.linspace(-0.4, 0.5, len(hinges), dtype=np.float32)
  root_pos = np.array([4.2, -3.1, 1.04], dtype=np.float32)
  root_quat = np.array([2**-0.5, 0.0, 0.0, 2**-0.5], dtype=np.float32)
  root_lin_vel = np.array([0.6, -0.2, 0.08], dtype=np.float32)
  root_ang_vel = np.array([0.52, 0.12, 0.68], dtype=np.float32)
  motion_file = tmp_path_factory.mktemp("r00_rsi") / "motion.npz"
  np.savez(
    motion_file,
    fps=50.0,
    qpos=np.tile(np.r_[root_pos, root_quat, joint_pos], (2, 1)),
    qvel=np.tile(np.r_[root_lin_vel, root_ang_vel, joint_vel], (2, 1)),
    starts=np.array([0]),
    lengths=np.array([2]),
    joint_names=np.array(names),
  )
  return SimpleNamespace(
    motion_file=str(motion_file),
    joint_names=names,
    joint_pos=torch.from_numpy(joint_pos),
    joint_vel=torch.from_numpy(joint_vel),
    root_quat=torch.from_numpy(root_quat),
    height=float(root_pos[2]),
    linear_velocity=torch.from_numpy(root_lin_vel),
    angular_velocity=torch.from_numpy(root_ang_vel),
  )


def _env(reference, probability, num_envs):
  cfg = r00_velocity_env_cfg(amp=True)
  cfg.events[REFERENCE_STATE_RESET_TERM] = reference_state_reset_event(
    reference.motion_file, probability
  )
  # Make the independently sampled command different from the reference twist.
  cfg.commands["twist"] = replace(
    cfg.commands["twist"],
    lin_vel_x=(0.3, 0.3),
    lin_vel_y=(-0.25, -0.25),
    ang_vel_z=(0.4, 0.4),
    standing_fraction=0.0,
    single_axis_fraction=0.0,
  )
  return create_env(cfg, num_envs=num_envs, device="cpu")


def _assert_bounded(actual, center, half_width):
  assert ((actual - center).abs() <= actual.new_tensor(half_width) + 1e-6).all()


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_r00_rsi_preserves_nominal_rows_and_uses_reference_state(
  reference, probability
):
  env = _env(reference, probability, num_envs=12)
  try:
    raw = env.unwrapped
    robot = raw.scene["robot"]
    obs, _ = raw.reset(seed=27)
    pos, quat, joint_pos, linear, angular, joint_vel = decompose_state(
      obs["full_state"]
    )
    using_reference = pos[:, 2] > 1.0
    if probability == 0.0:
      assert not using_reference.any()
    elif probability == 1.0:
      assert using_reference.all()
    else:
      assert using_reference.any() and not using_reference.all()

    nominal = ~using_reference
    if nominal.any():
      _assert_bounded(pos[nominal, 2], 0.96, 0.01)
      _assert_bounded(joint_pos[nominal], robot.data.default_joint_pos[nominal], 0.02)
      assert not linear[nominal].count_nonzero()
      assert not angular[nominal].count_nonzero()
      assert not joint_vel[nominal].count_nonzero()

    if using_reference.any():
      selected_pose = pos[using_reference]
      _assert_bounded(
        selected_pose,
        selected_pose.new_tensor([0.0, 0.0, reference.height]),
        (0.02, 0.02, 0.01),
      )
      assert raw.scene.env_origins[using_reference, :2].abs().max() > 0
      torch.testing.assert_close(
        robot.data.root_link_pos_w[using_reference],
        selected_pose + raw.scene.env_origins[using_reference],
      )
      selected_quat = quat[using_reference]
      reference_quat = reference.root_quat.expand_as(selected_quat)
      delta_quat = quat_mul(selected_quat, quat_conjugate(reference_quat))
      euler = torch.stack(euler_xyz_from_quat(delta_quat), dim=-1)
      _assert_bounded(euler, 0.0, (0.05, 0.05, 0.1))
      _assert_bounded(
        linear[using_reference], reference.linear_velocity, (0.1, 0.1, 0.05)
      )
      _assert_bounded(angular[using_reference], reference.angular_velocity, 0.1)
      torch.testing.assert_close(
        robot.data.root_link_ang_vel_b[using_reference],
        quat_apply_inverse(selected_quat, angular[using_reference]),
      )
      order = [reference.joint_names.index(name) for name in robot.joint_names]
      _assert_bounded(joint_pos[using_reference], reference.joint_pos[order], 0.02)
      torch.testing.assert_close(
        joint_vel[using_reference],
        reference.joint_vel[order].expand_as(joint_vel[using_reference]),
      )

    expected_command = torch.tensor([0.3, -0.25, 0.4]).expand_as(obs["goal"])
    torch.testing.assert_close(obs["goal"], expected_command)
  finally:
    env.close()


def test_r00_rsi_partial_reset_leaves_other_rows_untouched(reference):
  env = _env(reference, probability=1.0, num_envs=4)
  try:
    raw = env.unwrapped
    before, _ = raw.reset(seed=33)
    before_state = before["full_state"].clone()
    rows = torch.tensor([3, 1])
    after, _ = raw.reset(seed=9, env_ids=rows)
    torch.testing.assert_close(after["full_state"][[0, 2]], before_state[[0, 2]])
    assert not torch.equal(after["full_state"][rows], before_state[rows])
    _assert_bounded(after["full_state"][rows, 2], reference.height, 0.01)
  finally:
    env.close()
