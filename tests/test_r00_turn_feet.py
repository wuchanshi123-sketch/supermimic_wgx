"""Check turn foot geometry against the native MuJoCo model."""

import mujoco
import numpy as np
import torch

from supermimic.scripts.train_utils import create_env
from supermimic.tasks.r00_velocity_env_cfg import r00_velocity_env_cfg
from supermimic.utils.gencoord_helpers import compute_gencoord_state, decompose_state
from supermimic.utils.r00_turn_feet import R00TurnFeet
from supermimic.utils.velocity_angmom import VelocityAngularMomentum


def test_r00_turn_foot_geometry_matches_native_cylinders_and_link_rates():
  env = create_env(r00_velocity_env_cfg(), num_envs=2, device="cpu")
  try:
    raw = env.unwrapped
    robot = raw.scene["robot"]
    model = raw.sim.mj_model
    kinematics = VelocityAngularMomentum(raw)
    feet = R00TurnFeet(raw, kinematics)
    body_ids = robot.indexing.body_ids.cpu().numpy()
    root = int(body_ids[0])
    ankle_ids = [
      int(body_ids[robot.body_names.index(f"{side}_ankle_roll_link")])
      for side in ("left", "right")
    ]
    joint_qpos = robot.indexing.joint_q_adr.cpu().numpy()
    joint_qvel = robot.indexing.joint_v_adr.cpu().numpy()
    root_joint = model.body_jntadr[root]
    root_qpos = model.jnt_qposadr[root_joint]
    root_qvel = model.jnt_dofadr[root_joint]
    native = mujoco.MjData(model)
    # Nonlevel base and asymmetrical leg poses and angular velocities exercise
    # the root transform, cylinder end caps and L/R ordering.
    native.qpos[root_qpos : root_qpos + 3] = (0.2, -0.1, 1.02)
    orientation = np.array([1.0, 0.09, -0.13, 0.33])
    native.qpos[root_qpos + 3 : root_qpos + 7] = orientation / np.linalg.norm(
      orientation
    )
    native.qpos[joint_qpos] = np.linspace(-0.15, 0.18, 31)
    native.qvel[root_qvel + 3 : root_qvel + 6] = (0.3, -0.8, 1.2)
    native.qvel[joint_qvel] = np.linspace(-2.4, 1.3, 31)
    mujoco.mj_forward(model, native)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, native, mujoco.mjtObj.mjOBJ_BODY, root, velocity, 0)
    state = torch.tensor(
      np.concatenate(
        (
          native.xpos[root],
          native.xquat[root],
          native.qpos[joint_qpos],
          velocity[3:],
          velocity[:3],
          native.qvel[joint_qvel],
        )
      ),
      dtype=torch.float32,
    ).unsqueeze(0)
    rep = compute_gencoord_state(state).requires_grad_(True)
    h, yaw, _, _ = feet.from_kinematics(
      kinematics.body_kinematics(rep), state[:, :3], state[:, 3:7]
    )
    expected_h, expected_yaw = [], []
    for ankle in ankle_ids:
      lowest = []
      for i in np.flatnonzero(model.geom_bodyid == ankle):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
          axis = native.geom_xmat[i].reshape(3, 3)[:, 2]
          radius, half_length = model.geom_size[i, :2]
          lowest.append(
            native.geom_xpos[i, 2]
            - abs(axis[2]) * half_length
            - np.linalg.norm(axis[:2]) * radius
          )
      expected_h.append(min(lowest))
      mujoco.mj_objectVelocity(
        model, native, mujoco.mjtObj.mjOBJ_BODY, ankle, velocity, 0
      )
      expected_yaw.append(velocity[2])
    torch.testing.assert_close(h, torch.tensor([expected_h], dtype=torch.float32))
    torch.testing.assert_close(yaw, torch.tensor([expected_yaw], dtype=torch.float32))
    (grad,) = torch.autograd.grad(h.sum() + yaw.sum(), rep)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0

    obs, _ = raw.reset()
    pos, quat, *_ = decompose_state(obs["full_state"])
    predicted = feet.from_kinematics(
      kinematics.body_kinematics(compute_gencoord_state(obs["full_state"])), pos, quat
    )
    measured = feet.from_sim()
    torch.testing.assert_close(measured[0], predicted[0], atol=2e-6, rtol=0)
    torch.testing.assert_close(measured[1], predicted[1], atol=2e-6, rtol=0)
    assert measured[2].shape == measured[3].shape == (2, 2)
    assert torch.all((measured[3] >= 0) & (measured[3] <= 1))
  finally:
    env.close()
