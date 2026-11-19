"""Drive a local G1 velocity actor with a gamepad in native MuJoCo."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict import TensorDict

from supermimic.runners.velocity_runner import VelocityActor, VelocityRunnerCfg
from supermimic.scripts.native_viewer import ManagedNativeMujocoViewer
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.tasks.velocity_env_cfg import VelocityCommand, velocity_env_cfg
from supermimic.utils.gamepad import Gamepad
from supermimic.utils.local_checkpoints import resolve_checkpoint_actor


@dataclass
class VelocityPlayConfig:
  checkpoint: Path
  """Local velocity-policy actor.pt. A run-directory symlink is supported."""
  gamepad_device: Path
  """Linux joystick device or /dev/input/by-id joystick symlink."""
  device: str = "cpu"
  runner_config: Path | None = None
  """Optional runner_config.json. Defaults to the checkpoint's sibling file."""
  gamepad_deadzone: float = 0.1
  gamepad_axes: tuple[int, int, int] | None = None
  """Override raw left X, left Y, and right X axis indices."""


def load_velocity_actor(
  checkpoint: Path,
  sensor_dim: int,
  num_actions: int,
  command_dim: int,
  device: str,
  runner_config: Path | None = None,
) -> tuple[VelocityActor, int]:
  checkpoint = resolve_checkpoint_actor(checkpoint.expanduser())
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Actor checkpoint not found: {checkpoint}")

  action_limit = VelocityRunnerCfg.action_limit
  config_path = runner_config or checkpoint.with_name("runner_config.json")
  config_path = config_path.expanduser()
  if runner_config is not None or config_path.is_file():
    try:
      raw = json.loads(config_path.read_text(encoding="utf-8"))["runner_cfg"]
      action_limit = float(raw["action_limit"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
      raise ValueError(
        f"Could not read the velocity actor config from {config_path}: {exc}"
      ) from exc
  else:
    print(
      f"[INFO] Using the default action limit {action_limit:g} (no runner_config.json)"
    )

  state = torch.load(checkpoint, map_location="cpu", weights_only=True)
  if not isinstance(state, dict):
    raise ValueError(f"Expected a velocity actor state dict in {checkpoint}")

  # Linear weights encode the history and hidden widths, even for a copied actor.pt.
  layers = []
  for name, tensor in state.items():
    match = re.fullmatch(r"network\.(\d+)\.weight", name)
    if match and isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
      layers.append((int(match[1]), tensor))
  layers.sort(key=lambda layer: layer[0])
  if not layers or layers[-1][1].shape[0] != num_actions:
    raise ValueError(f"Checkpoint does not match the G1 velocity actor: {checkpoint}")
  input_dim = layers[0][1].shape[1]
  history_length, extra = divmod(input_dim - command_dim, sensor_dim)
  if history_length < 1 or extra:
    raise ValueError(
      f"Checkpoint has an incompatible proprioception history: {checkpoint}"
    )

  cfg = VelocityRunnerCfg(
    actor_hidden_dims=tuple(weight.shape[0] for _, weight in layers[:-1]),
    history_length=history_length,
    buffer_size=max(VelocityRunnerCfg.buffer_size, history_length),
    action_limit=action_limit,
    exploration_noise=0.0,
  )
  actor = VelocityActor(input_dim, num_actions, cfg)
  try:
    actor.load_state_dict(state)
  except RuntimeError as exc:
    raise ValueError(
      f"Checkpoint does not match the G1 velocity actor: {checkpoint}"
    ) from exc
  return actor.to(device).eval(), history_length


class GamepadVelocityPolicy:
  """Build the actor's training-time history layout around a live command."""

  def __init__(
    self,
    actor: VelocityActor,
    history_length: int,
    command: VelocityCommand,
    gamepad: Gamepad,
  ) -> None:
    self.actor = actor
    self.history_length = history_length
    self.gamepad = gamepad
    self._history: torch.Tensor | None = None
    self._command = command.command
    ranges = (command.cfg.lin_vel_x, command.cfg.lin_vel_y, command.cfg.ang_vel_z)
    self._positive = self._command.new_tensor([max(hi, 0.0) for _, hi in ranges])
    self._negative = self._command.new_tensor([max(-lo, 0.0) for lo, _ in ranges])

  def __call__(self, obs: TensorDict) -> torch.Tensor:
    sensor = obs["actor"].unsqueeze(1)
    if self._history is None:
      self._history = sensor.repeat(1, self.history_length, 1)
    else:
      self._history = torch.cat([self._history[:, 1:], sensor], dim=1)

    axes = self._command.new_tensor(self.gamepad.read())
    self._command.copy_(axes * torch.where(axes >= 0, self._positive, self._negative))
    actor_input = torch.cat([self._history.flatten(1), self._command], dim=-1)
    return self.actor(TensorDict({"actor": actor_input}, obs.batch_size))

  def reset(self, dones: torch.Tensor | None = None) -> None:
    # This player presents a single simulated robot.
    if dones is None or bool(dones.any()):
      self._history = None


def play_velocity(cfg: VelocityPlayConfig) -> None:
  device = setup_device(cfg.device)
  env_cfg = velocity_env_cfg(play=True)
  env_cfg.sim.mujoco.iterations = 50
  env_cfg.sim.mujoco.ls_iterations = 50

  with Gamepad(
    cfg.gamepad_device, cfg.gamepad_deadzone, axes=cfg.gamepad_axes
  ) as gamepad:
    env = create_env(env_cfg, num_envs=1, device=device)
    try:
      obs = env.get_observations()
      command = env.unwrapped.command_manager.get_term("twist")
      assert isinstance(command, VelocityCommand)
      actor, history_length = load_velocity_actor(
        cfg.checkpoint,
        obs["actor"].shape[-1],
        env.num_actions,
        command.command.shape[-1],
        device,
        cfg.runner_config,
      )
      policy = GamepadVelocityPolicy(actor, history_length, command, gamepad)
      print(f"[INFO] Loaded velocity actor: {cfg.checkpoint.expanduser()}", flush=True)
      print(
        f"[INFO] Gamepad: {gamepad.device} | Space: pause | Enter: reset", flush=True
      )
      ManagedNativeMujocoViewer(env, policy, enable_perturbations=False).run()
    finally:
      env.close()
