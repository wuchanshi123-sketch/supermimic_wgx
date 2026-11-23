"""Differentiable torso stability signals for velocity control."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from supermimic.utils.gencoord_helpers import decompose_rep_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class VelocityTorsoKinematics:
  """Recover torso tilt and angular speed from the root and waist state."""

  def __init__(
    self, env: ManagerBasedRlEnv, torso_body_name: str = "torso_link"
  ) -> None:
    robot = env.scene["robot"]
    model = env.sim.mj_model
    body_ids = robot.indexing.body_ids.cpu().tolist()
    body_id = body_ids[robot.body_names.index(torso_body_name)]
    joint_cols = {
      address: column
      for column, address in enumerate(robot.indexing.joint_q_adr.cpu().tolist())
    }
    hinges = []
    while body_id != body_ids[0]:
      joint = int(model.body_jntadr[body_id])
      if (
        model.body_jntnum[body_id] != 1
        or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
      ):
        raise ValueError(
          f"Expected one hinge per body between the robot root and {torso_body_name}"
        )
      address = int(model.jnt_qposadr[joint])
      fixed_q = model.body_quat[body_id]
      local_axis = model.jnt_axis[joint]
      parent_axis, turn_q = np.empty(3), np.empty(4)
      mujoco.mju_rotVecQuat(parent_axis, local_axis, fixed_q)
      mujoco.mju_mulQuat(turn_q, fixed_q, np.r_[0.0, local_axis])
      hinges.append(
        (
          joint_cols[address],
          float(model.qpos0[address]),
          torch.tensor(parent_axis, device=env.device, dtype=torch.float32),
          torch.tensor(fixed_q, device=env.device, dtype=torch.float32),
          torch.tensor(turn_q, device=env.device, dtype=torch.float32),
        )
      )
      body_id = int(model.body_parentid[body_id])
    self._hinges = tuple(reversed(hinges))

  def __call__(
    self, rep_state: torch.Tensor, root_quat_w: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 1 - cos(tilt) and squared world-horizontal angular speed."""
    _, _, _, root_ang_vel_b, joint_pos, joint_vel = decompose_rep_state(rep_state)
    torso_quat_w = root_quat_w
    torso_ang_vel_w = quat_apply(root_quat_w, root_ang_vel_b)
    # Walk the short waist chain in parent order. Each hinge adds its speed
    # along the world-frame axis before rotating its child's frame.
    for column, qpos0, parent_axis, fixed_q, turn_q in self._hinges:
      axis_w = quat_apply(torso_quat_w, parent_axis.expand_as(torso_ang_vel_w))
      torso_ang_vel_w = torso_ang_vel_w + axis_w * joint_vel[..., column : column + 1]
      half_angle = (joint_pos[..., column : column + 1] - qpos0) * 0.5
      hinge_q = half_angle.cos() * fixed_q + half_angle.sin() * turn_q
      torso_quat_w = quat_mul(torso_quat_w, hinge_q)

    tilt_error = 2.0 * torso_quat_w[..., 1:3].square().sum(dim=-1)
    roll_pitch_speed_sq = torso_ang_vel_w[..., :2].square().sum(dim=-1)
    return tilt_error, roll_pitch_speed_sq
