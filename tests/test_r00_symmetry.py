"""Check R00 sagittal reflection against its native MuJoCo joint axes and FK."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.utils.r00_symmetry import R00Symmetry


@pytest.fixture(scope="module")
def r00_model():
  path = (
    Path(__file__).resolve().parents[1] / "src/supermimic/assets/r00/r00_serial.xml"
  )
  return mujoco.MjModel.from_xml_path(str(path))


def test_mirror_action_matches_native_link_poses_in_shuffled_joint_order(r00_model):
  model = r00_model
  rng = np.random.default_rng(61)
  joint_ids = rng.permutation(np.arange(1, model.njnt))
  names = [model.joint(int(j)).name for j in joint_ids]
  symmetry = R00Symmetry(names, 1, "cpu")
  source, mirrored = mujoco.MjData(model), mujoco.MjData(model)
  reflection = np.diag([1.0, -1.0, 1.0])
  source.qpos[:3] = [0.25, -0.6, 0.95]
  source.qpos[3:7] = [1.0, 0.15, -0.2, 0.24]
  source.qpos[3:7] /= np.linalg.norm(source.qpos[3:7])
  mirrored.qpos[:3] = reflection @ source.qpos[:3]
  mirrored.qpos[3:7] = source.qpos[3:7] * [1.0, -1.0, 1.0, -1.0]
  columns = model.jnt_qposadr[joint_ids]
  source.qpos[columns] = rng.uniform(*model.jnt_range[joint_ids].T)
  mirrored.qpos[columns] = symmetry.mirror_action(
    torch.from_numpy(source.qpos[columns])
  ).numpy()

  mujoco.mj_kinematics(model, source)
  mujoco.mj_kinematics(model, mirrored)
  for jid, name in zip(joint_ids, names, strict=True):
    other = (
      name.replace("left_", "right_")
      if name.startswith("left_")
      else name.replace("right_", "left_")
    )
    dest_body = model.jnt_bodyid[jid]
    src_body = model.jnt_bodyid[model.joint(other).id]
    np.testing.assert_allclose(
      mirrored.xpos[dest_body], reflection @ source.xpos[src_body], atol=2e-6
    )
    np.testing.assert_allclose(
      mirrored.xmat[dest_body].reshape(3, 3),
      reflection @ source.xmat[src_body].reshape(3, 3) @ reflection,
      atol=2e-6,
    )


@pytest.mark.parametrize("history_length", [1, 3])
def test_actor_mirror_keeps_history_order_and_transforms_live_command(
  r00_model, history_length
):
  names = [f"robot/{r00_model.joint(i).name}" for i in range(1, r00_model.njnt)]
  symmetry = R00Symmetry(names, history_length, "cpu")
  n = len(names)
  width = 6 + 3 * n
  obs = torch.arange(2 * (history_length * width + 3), dtype=torch.float64).reshape(
    2, -1
  )
  obs.requires_grad_()
  reflected = symmetry.mirror_obs(obs)
  torch.testing.assert_close(reflected[:, -3:], obs[:, -3:] * torch.tensor([1, -1, -1]))
  for source, mirror in zip(
    obs[:, :-3].reshape(2, history_length, width).unbind(1),
    reflected[:, :-3].reshape(2, history_length, width).unbind(1),
    strict=True,
  ):
    torch.testing.assert_close(
      mirror[:, :6], source[:, :6] * torch.tensor([1, -1, 1, -1, 1, -1])
    )
    for offset in (6, 6 + n, 6 + 2 * n):
      torch.testing.assert_close(
        mirror[:, offset : offset + n],
        symmetry.mirror_action(source[:, offset : offset + n]),
      )
  torch.testing.assert_close(symmetry.mirror_obs(reflected), obs)
  weight = torch.linspace(-0.4, 1.0, obs.numel(), dtype=obs.dtype).reshape_as(obs)
  (gradient,) = torch.autograd.grad((reflected * weight).sum(), obs)
  torch.testing.assert_close(gradient, symmetry.mirror_obs(weight))


def test_mirror_rejects_incompatible_input():
  with pytest.raises(ValueError, match="unpaired"):
    R00Symmetry(["left_elbow_pitch_joint"], 1, "cpu")
  symmetry = R00Symmetry(["neck_pitch_joint"], 1, "cpu")
  with pytest.raises(ValueError, match="action width"):
    symmetry.mirror_action(torch.zeros(2, 5))
  with pytest.raises(ValueError, match="observation width"):
    symmetry.mirror_obs(torch.zeros(2, 42))
