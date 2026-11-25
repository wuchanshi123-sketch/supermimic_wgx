"""Export a standalone velocity actor to ONNX for CPU deployment."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import torch
from onnx.reference import ReferenceEvaluator
from torch import nn

from supermimic.runners.exporter import _export_onnx
from supermimic.runners.velocity_runner import VelocityActor
from supermimic.scripts.play_velocity import load_velocity_actor


class _VelocityOnnxActor(nn.Module):
  def __init__(self, actor: VelocityActor) -> None:
    super().__init__()
    self.actor = actor

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.actor({"actor": obs})


def export_velocity_onnx(
  checkpoint: Path,
  output: Path | None = None,
  *,
  runner_config: Path | None = None,
  overwrite: bool = False,
) -> Path:
  """Write a validated single-batch actor graph and return its path."""
  checkpoint = checkpoint.expanduser()
  if checkpoint.is_dir():
    checkpoint /= "actor.pt"
  # Pin a run-directory alias before loading weights and its sibling runner JSON.
  checkpoint = checkpoint.resolve(strict=True)
  output = (
    output.expanduser() if output is not None else checkpoint.with_suffix(".onnx")
  )
  output = output.parent.resolve() / output.name
  if output.suffix != ".onnx":
    raise ValueError(f"Output path must end in .onnx: {output}")
  if not overwrite and (output.exists() or output.is_symlink()):
    raise FileExistsError(
      f"ONNX file already exists: {output}. Use --overwrite to replace it."
    )

  actor, history_length = load_velocity_actor(checkpoint, runner_config=runner_config)
  wrapper = _VelocityOnnxActor(actor).eval()
  mean, std = actor.normalizer._mean, actor.normalizer._std
  example = torch.zeros_like(mean)
  varied = mean + std * torch.linspace(-1.0, 1.0, mean.numel()).reshape_as(mean)
  output.parent.mkdir(parents=True, exist_ok=True)
  with TemporaryDirectory(dir=output.parent, prefix=".velocity-onnx-") as directory:
    staged = Path(directory) / output.name
    _export_onnx(
      wrapper, (example,), ["obs"], ["action"], str(staged), opset_version=17
    )
    model = onnx.load(staged)
    onnx.helper.set_model_props(
      model,
      {
        "history_length": str(history_length),
        "raw_action_limit": str(actor.action_limit),
        "observation_layout": "frame-major proprioception, then current vx, vy, wz",
      },
    )
    onnx.checker.check_model(model, full_check=True)
    evaluator = ReferenceEvaluator(model)
    with torch.inference_mode():
      for obs in (example, varied):
        (action,) = evaluator.run(["action"], {"obs": obs.numpy()})
        torch.testing.assert_close(
          torch.from_numpy(action), wrapper(obs), atol=1e-5, rtol=1e-4
        )

    onnx.save(model, staged, save_as_external_data=False)
    if overwrite:
      staged.replace(output)
    else:
      output.hardlink_to(staged)

  input_width = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
  output_width = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
  print(
    f"[INFO] Exported {output} (obs [1, {input_width}] -> action [1, {output_width}])"
  )
  return output


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--checkpoint", type=Path, required=True, help="actor.pt or its directory"
  )
  parser.add_argument(
    "--output", type=Path, help="ONNX path, defaulting to the input with a .onnx suffix"
  )
  parser.add_argument(
    "--runner-config", type=Path, help="Override the checkpoint's runner_config.json"
  )
  parser.add_argument(
    "--overwrite", action="store_true", help="Replace an existing ONNX after validation"
  )
  args = parser.parse_args()
  try:
    export_velocity_onnx(
      args.checkpoint,
      args.output,
      runner_config=args.runner_config,
      overwrite=args.overwrite,
    )
  except (OSError, ValueError) as exc:
    parser.error(str(exc))


if __name__ == "__main__":
  main()
