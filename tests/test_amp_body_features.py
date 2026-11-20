"""Compare AMP body features with MuJoCo link kinematics on G1."""

from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from supermimic.utils.amp_body_features import (
  AMP_ANCHOR_NAME,
  AMP_BODY_NAMES,
  AmpBodyFeatures,
)
from supermimic.utils.gencoord_helpers import compute_gencoord_state
from supermimic.utils.velocity_angmom import VelocityAngularMomentum


def test_amp_features_match_g1_link_poses_and_twists():
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  body_ids = torch.arange(1, model.nbody)
  robot = SimpleNamespace(
    indexing=SimpleNamespace(body_ids=body_ids, joint_q_adr=torch.arange(7, model.nq)),
    body_names=tuple(model.body(i).name for i in body_ids.tolist()),
  )
  env = SimpleNamespace(
    scene={"robot": robot},
    sim=SimpleNamespace(mj_model=model),
    cfg=SimpleNamespace(events={}),
    device="cpu",
  )
  kinematics = VelocityAngularMomentum(env)
  extract = AmpBodyFeatures(kinematics)
  link_ids = [model.body(name).id for name in AMP_BODY_NAMES]
  torso_id = model.body(AMP_ANCHOR_NAME).id
  data = mujoco.MjData(model)
  rng = np.random.default_rng(37)
  states, expected = [], []
  for _ in range(6):
    data.qpos[:] = rng.normal(size=model.nq)
    data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
    data.qvel[:] = rng.normal(size=model.nv)
    mujoco.mj_forward(model, data)
    world_rotations = data.xmat.reshape(-1, 3, 3)
    states.append(
      np.r_[
        data.qpos, data.qvel[:3], world_rotations[1] @ data.qvel[3:6], data.qvel[6:]
      ]
    )

    torso_t = world_rotations[torso_id].T
    rel_pos = (data.xpos[link_ids] - data.xpos[torso_id]) @ torso_t.T
    rel_rot = (torso_t @ world_rotations[link_ids])[..., :2]
    twist = np.zeros((len(link_ids), 6))
    for i, link_id in enumerate(link_ids):
      mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_XBODY, link_id, twist[i], 1
      )
    expected.append(
      np.r_[
        rel_pos.ravel(), rel_rot.ravel(), twist[:, 3:].ravel(), twist[:, :3].ravel()
      ]
    )

  state = compute_gencoord_state(torch.tensor(np.stack(states), dtype=torch.float32))
  state = state.reshape(2, 3, -1).requires_grad_()
  result = extract(state, kinematics.body_kinematics(state))
  torch.testing.assert_close(result, extract(state))
  torch.testing.assert_close(
    result,
    torch.tensor(np.stack(expected), dtype=torch.float32).reshape(2, 3, 195),
    atol=5e-6,
    rtol=1e-5,
  )
  (grad,) = torch.autograd.grad(result.square().sum(), state)
  assert torch.isfinite(grad).all()
  assert torch.count_nonzero(grad[..., 4:7]) > 0  # Pelvis translation influences AMP.
  assert torch.count_nonzero(grad[..., 7:10]) > 0
  assert torch.count_nonzero(grad[..., 10:]) > 0
