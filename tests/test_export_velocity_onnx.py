"""Check the standalone ONNX contract against the checkpoint's PyTorch actor."""

import json

import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from supermimic.runners.velocity_runner import VelocityActor, VelocityRunnerCfg
from supermimic.scripts.export_velocity_onnx import export_velocity_onnx
from supermimic.utils.local_checkpoints import update_latest_checkpoint


@pytest.mark.parametrize("num_actions,history", [(31, 1), (29, 2)])
def test_export_velocity_actor_with_saved_normalizer(tmp_path, num_actions, history):
  input_dim = (6 + 3 * num_actions) * history + 3
  actor = VelocityActor(
    input_dim,
    num_actions,
    VelocityRunnerCfg(actor_hidden_dims=(16, 8), action_limit=1.4),
  )
  actor.normalizer.update(torch.linspace(-2, 3, 4 * input_dim).reshape(4, input_dim))
  actor.eval()
  checkpoint = tmp_path / "checkpoint_2000"
  checkpoint.mkdir()
  torch.save(actor.state_dict(), checkpoint / "actor.pt")
  (checkpoint / "runner_config.json").write_text(
    json.dumps({"runner_cfg": {"action_limit": 1.4}})
  )
  update_latest_checkpoint(checkpoint)

  path = export_velocity_onnx(tmp_path)
  assert path == checkpoint / "actor.onnx"
  model = onnx.load(path)
  meta = {item.key: item.value for item in model.metadata_props}
  assert meta["history_length"] == str(history)
  assert meta["raw_action_limit"] == "1.4"
  obs = torch.linspace(0.2, -0.7, input_dim).reshape(1, input_dim)
  with torch.inference_mode():
    expected = actor({"actor": obs})
  assert expected.shape == (1, num_actions)
  (action,) = ReferenceEvaluator(model).run(["action"], {"obs": obs.numpy()})
  torch.testing.assert_close(torch.from_numpy(action), expected, atol=1e-6, rtol=1e-5)

  with pytest.raises(FileExistsError, match="--overwrite"):
    export_velocity_onnx(checkpoint)
