"""Verify the differentiable momentum signal against MuJoCo subtree dynamics."""

from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from supermimic.utils.velocity_angmom import VelocityAngularMomentum


def test_subtree_angular_momentum_with_randomized_com():
  model = mujoco.MjModel.from_xml_string(
    """
    <mujoco>
      <worldbody>
        <body name="pelvis">
          <freejoint/>
          <geom type="ellipsoid" size="0.1 0.15 0.12" mass="10"/>
          <body name="torso_link" pos="0.06 0.06 0.1" quat="0.6 0 0 0.8">
            <joint type="hinge" axis="1 0 0" pos="0.02 0.01 -0.02"/>
            <geom type="box" size="0.15 0.16 0.11" mass="7" pos="0.03 0.05 0"/>
            <body name="fixed_link" pos="-0.03 0.01 0.15">
              <geom size="0.09" mass="0.8"/>
            </body>
          </body>
          <body name="leg" pos="0.01 -0.05 -0.1" quat="0.6 0 0.8 0">
            <joint type="hinge" axis="0 0 1"/>
            <geom type="capsule" size="0.04" fromto="0 0 0 0.1 0 -0.13" mass="2"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
  )
  indexing = SimpleNamespace(
    body_ids=torch.arange(1, 5), joint_q_adr=torch.tensor([7, 8])
  )
  robot = SimpleNamespace(
    indexing=indexing, body_names=[model.body(i).name for i in range(1, 5)]
  )
  env = SimpleNamespace(
    scene={"robot": robot},
    sim=SimpleNamespace(mj_model=model),
    cfg=SimpleNamespace(events={}),
    device="cpu",
  )
  momentum = VelocityAngularMomentum(env)
  rng = np.random.default_rng(359)
  data = mujoco.MjData(model)
  features = []
  torso_positions = []
  expected = []
  for _ in range(10):
    data.qpos[:] = rng.standard_normal(model.nq)
    data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
    data.qvel[:] = rng.standard_normal(model.nv)
    torso_ipos = rng.uniform(-0.08, 0.08, 3)
    model.body_ipos[momentum.com_body_id] = torso_ipos
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)
    expected.append(np.square(data.subtree_angmom[1]).sum())
    features.append(np.r_[np.zeros(7), data.qvel[3:6], data.qpos[7:], data.qvel[6:]])
    torso_positions.append(torso_ipos)

  states = torch.tensor(np.stack(features), dtype=torch.float32, requires_grad=True)
  offsets = torch.tensor(
    np.stack(torso_positions), dtype=torch.float32, requires_grad=True
  )
  result = momentum(states, offsets)
  torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float32))
  grads = torch.autograd.grad(result.sum(), (states, offsets))
  assert all(torch.isfinite(grad).all() and grad.abs().max() > 0 for grad in grads)
