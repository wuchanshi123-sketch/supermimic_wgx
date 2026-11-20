import numpy as np
import pytest
import torch

from supermimic.utils.amp_reference import AmpReference


def _write_clip(path, start, *, length=3, fps=50.0, joints=("hip", "knee")):
  frames = np.arange(start, start + length, dtype=np.float32)
  root_pos = np.zeros((length, 3), dtype=np.float32)
  root_pos[:, 0] = frames
  np.savez(
    path,
    fps=np.array([fps]),
    joint_names=np.array(joints),
    root_pos_w=root_pos,
    root_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (length, 1)),
    root_lin_vel_w=np.zeros((length, 3)),
    root_ang_vel_w=np.zeros((length, 3)),
    joint_pos=np.stack((frames + 10.0, frames + 20.0), axis=-1),
    joint_vel=np.stack((frames + 30.0, frames + 40.0), axis=-1),
  )


@pytest.mark.parametrize("source_kind", ["file", "directory", "glob"])
def test_amp_reference_reorders_joints_and_never_crosses_clips(tmp_path, source_kind):
  first, second = tmp_path / "a.npz", tmp_path / "b.npz"
  _write_clip(first, 0)
  _write_clip(second, 100)
  source = {"file": first, "directory": tmp_path, "glob": str(tmp_path / "*.npz")}
  reference = AmpReference.load(
    source[source_kind], ("robot/knee", "robot/hip"), 0.02, "cpu"
  )

  expected_starts = [0, 1] if source_kind == "file" else [0, 1, 3, 4]
  assert reference.transition_starts.tolist() == expected_starts
  assert reference.num_frames == (3 if source_kind == "file" else 6)
  assert reference.num_clips == (1 if source_kind == "file" else 2)
  sampled = reference.sample(32)
  assert sampled.shape == (32, 2, 17)
  assert sampled.dtype == torch.float32
  torch.testing.assert_close(sampled[:, 1, 0] - sampled[:, 0, 0], torch.ones(32))
  torch.testing.assert_close(sampled[..., 7], sampled[..., 0] + 20.0)
  torch.testing.assert_close(sampled[..., 8], sampled[..., 0] + 10.0)
  torch.testing.assert_close(sampled[..., -2], sampled[..., 0] + 40.0)
  torch.testing.assert_close(sampled[..., -1], sampled[..., 0] + 30.0)


def test_amp_reference_rejects_wrong_fps_or_joint_names(tmp_path):
  motion = tmp_path / "motion.npz"
  _write_clip(motion, 0, fps=25.0)
  with pytest.raises(ValueError, match=r"motion\.npz: expected 50 Hz.*got 25 Hz"):
    AmpReference.load(motion, ("hip", "knee"), 0.02, "cpu")

  _write_clip(motion, 0, joints=("hip", "hip"))
  with pytest.raises(ValueError, match="joint_names must match"):
    AmpReference.load(motion, ("hip", "knee"), 0.02, "cpu")


def test_amp_reference_samples_unequal_clips_uniformly(tmp_path):
  _write_clip(tmp_path / "short.npz", 100, length=2)
  _write_clip(tmp_path / "long.npz", 0, length=41)
  reference = AmpReference.load(tmp_path, ("hip", "knee"), 0.02, "cpu")

  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(8)
    starts = reference.sample_indices(8192)

  assert reference.clip_starts.tolist() == [0, 41]
  assert reference.clip_lengths.tolist() == [41, 2]
  assert torch.isin(starts, reference.transition_starts).all()
  assert 0.48 < (starts == 41).float().mean() < 0.52
