"""Drive a local G1 velocity actor with a gamepad in native MuJoCo."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mujoco.viewer import Handle
from tensordict import TensorDict

from supermimic.representations import StateRepresentation
from supermimic.runners.supermimic_runner import WorldModel, build_world_model
from supermimic.runners.velocity_runner import VelocityActor, VelocityRunnerCfg
from supermimic.scripts.native_viewer import ManagedNativeMujocoViewer
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.scripts.viewers import create_wm_vis_model
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
  num_steps_rollout: int = 100
  """Reseed the world model ghost from the simulator after this many control steps."""


def _linear_weights(state: dict, module: str) -> list[torch.Tensor]:
  layers = []
  for name, tensor in state.items():
    match = re.fullmatch(rf"{module}\.(\d+)\.weight", name)
    if match and isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
      layers.append((int(match[1]), tensor))
  layers.sort(key=lambda layer: layer[0])
  return [tensor for _, tensor in layers]


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
  layers = _linear_weights(state, "network")
  if not layers or layers[-1].shape[0] != num_actions:
    raise ValueError(f"Checkpoint does not match the G1 velocity actor: {checkpoint}")
  input_dim = layers[0].shape[1]
  history_length, extra = divmod(input_dim - command_dim, sensor_dim)
  if history_length < 1 or extra:
    raise ValueError(
      f"Checkpoint has an incompatible proprioception history: {checkpoint}"
    )

  cfg = VelocityRunnerCfg(
    actor_hidden_dims=tuple(weight.shape[0] for weight in layers[:-1]),
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


def load_velocity_world_model(
  checkpoint: Path, env: RslRlVecEnvWrapper, device: str
) -> WorldModel:
  bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
  if not isinstance(bundle, dict) or not isinstance(bundle.get("dynamics"), dict):
    raise ValueError(f"Expected a G1 velocity world model bundle in {checkpoint}")
  layers = _linear_weights(bundle["dynamics"], "mlp")
  if not layers:
    raise ValueError(f"Missing dynamics network weights in {checkpoint}")
  cfg = VelocityRunnerCfg(dynamics_hidden_dims=tuple(w.shape[0] for w in layers[:-1]))
  wm = build_world_model(cfg, env, device, bind_representation=StateRepresentation.bind)
  try:
    wm.load_bundle(bundle)
  except (KeyError, RuntimeError) as exc:
    raise ValueError(
      f"Checkpoint does not match the G1 velocity world model: {checkpoint}"
    ) from exc
  wm.dynamics.eval()
  return wm


def _actor_observations(history: torch.Tensor, command: torch.Tensor) -> TensorDict:
  actor_input = torch.cat([history.flatten(1), command], dim=-1)
  return TensorDict({"actor": actor_input}, [history.shape[0]])


class VelocityWorldModelGhost:
  """Track a second, closed-loop robot using the learned dynamics."""

  def __init__(
    self,
    wm: WorldModel,
    env: ManagerBasedRlEnv,
    actor: VelocityActor,
    rollout_steps: int,
  ) -> None:
    if rollout_steps < 1:
      raise ValueError("--num-steps-rollout must be positive")
    self.wm = wm
    self.env = env
    self.actor = actor
    self.rollout_steps = rollout_steps
    self.model = create_wm_vis_model(env.sim.mj_model)
    robot = env.scene["robot"]
    self._root_indices = robot.indexing.free_joint_q_adr.cpu().numpy()
    self._joint_indices = robot.indexing.joint_q_adr.cpu().numpy()
    self._origin = env.scene.env_origins[0].cpu().numpy().copy()
    self._qpos_buffer = self.model.qpos0.copy()
    self._warned_nonfinite = False
    self.reset()

  def reset(self) -> None:
    self._state: tuple[torch.Tensor, torch.Tensor] | None = None
    self._steps = 0
    self.qpos: np.ndarray | None = None

  @torch.no_grad()
  def step(
    self,
    full_state: torch.Tensor,
    command: torch.Tensor,
    real_history: torch.Tensor,
    real_action: torch.Tensor,
  ) -> None:
    representation = self.wm.representation
    if self._state is None or self._steps >= self.rollout_steps:
      # Both robots start this rollout at the same pose and actor history.
      aux = representation.extract_aux_state(full_state)
      rep_state = representation.represent(full_state)
      self._sim_ctx = self.wm.sim_context.read(self.env)
      self._history = real_history
      action = real_action
      self._steps = 0
    else:
      aux, rep_state = self._state
      sensor = representation.compute_actor_sensor(
        rep_state,
        self._sim_ctx,
        aux_state=aux,
        joint_state=representation.joint_state(rep_state),
        last_action=self._last_action,
      )
      self._history = torch.cat([self._history[:, 1:], sensor.unsqueeze(1)], dim=1)
      action = self.actor(_actor_observations(self._history, command))

    next_aux, next_rep = self.wm.dynamics(rep_state, action, aux, self._sim_ctx)
    predicted = (
      representation.reconstruct_full_state(next_aux, next_rep)[0].cpu().numpy()
    )
    if not np.isfinite(predicted).all():
      self.reset()
      if not self._warned_nonfinite:
        print(
          "[WARN] The world model produced a non-finite state. Reseeding its ghost."
        )
        self._warned_nonfinite = True
      return

    self._state = (next_aux, next_rep)
    self._last_action = action
    self._steps += 1
    self._qpos_buffer[self._root_indices] = predicted[:7]
    self._qpos_buffer[self._root_indices[:3]] += self._origin
    self._qpos_buffer[self._joint_indices] = predicted[7 : 7 + len(self._joint_indices)]
    self.qpos = self._qpos_buffer

  def draw(self, visualizer: DebugVisualizer) -> None:
    if self.qpos is not None:
      visualizer.add_ghost_entity(self.qpos, model=self.model, label="wm_prediction")


class GamepadVelocityPolicy:
  """Build the actor's training-time history layout around a live command."""

  def __init__(
    self,
    actor: VelocityActor,
    history_length: int,
    command: VelocityCommand,
    gamepad: Gamepad,
    ghost: VelocityWorldModelGhost | None = None,
  ) -> None:
    self.actor = actor
    self.history_length = history_length
    self.gamepad = gamepad
    self.ghost = ghost
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
    action = self.actor(_actor_observations(self._history, self._command))
    if self.ghost is not None:
      self.ghost.step(obs["full_state"], self._command, self._history, action)
    return action

  def reset(self, dones: torch.Tensor | None = None) -> None:
    # This player presents a single simulated robot.
    if dones is None or bool(dones.any()):
      self._history = None
      if self.ghost is not None:
        self.ghost.reset()


class GamepadVelocityViewer(ManagedNativeMujocoViewer):
  """Show the scaled gamepad command alongside the normal playback status."""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    policy: GamepadVelocityPolicy,
    command: VelocityCommand,
  ) -> None:
    super().__init__(env, policy, enable_perturbations=False)
    self._velocity_command = command

  def _set_status_overlay(self, viewer: Handle) -> None:
    status = self.get_status()
    capped = " [CAPPED]" if status.capped else ""
    vx, vy, wz = self._velocity_command.command[self.env_idx].tolist()
    font = mujoco.mjtFontScale.mjFONTSCALE_150.value
    if not viewer.is_running():
      return
    viewer.set_texts(
      [
        (
          font,
          mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
          "Env\nStep\nStatus\nSpeed\nTarget RT\nActual RT",
          f"{self.env_idx + 1}/{self.env.num_envs}\n"
          f"{status.step_count}\n"
          f"{'PAUSED' if status.paused else 'RUNNING'}{capped}\n"
          f"{status.speed_label}\n"
          f"{status.target_realtime:.2f}x\n"
          f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)",
        ),
        (
          font,
          mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
          "Gamepad command\nvx (m/s)\nvy (m/s)\nwz (rad/s)",
          f" \n{vx:+.2f}\n{vy:+.2f}\n{wz:+.2f}",
        ),
      ]
    )


def play_velocity(cfg: VelocityPlayConfig, *, amp: bool = False) -> None:
  if cfg.num_steps_rollout < 1:
    raise ValueError("--num-steps-rollout must be positive")
  actor_path = resolve_checkpoint_actor(cfg.checkpoint.expanduser())
  wm_path = actor_path.with_name("world_model.pt")
  device = setup_device(cfg.device)
  env_cfg = velocity_env_cfg(play=True, amp=amp)
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
        actor_path,
        obs["actor"].shape[-1],
        env.num_actions,
        command.command.shape[-1],
        device,
        cfg.runner_config,
      )
      ghost = None
      if wm_path.is_file():
        wm = load_velocity_world_model(wm_path, env, device)
        ghost = VelocityWorldModelGhost(wm, env.unwrapped, actor, cfg.num_steps_rollout)
        original_visualizers = env.unwrapped.update_visualizers

        def update_visualizers(visualizer: DebugVisualizer) -> None:
          original_visualizers(visualizer)
          ghost.draw(visualizer)

        env.unwrapped.update_visualizers = update_visualizers
        print(f"[INFO] Loaded world model ghost: {wm_path}", flush=True)
      else:
        print(f"[INFO] Playing actor only (no world_model.pt at {wm_path})", flush=True)

      policy = GamepadVelocityPolicy(actor, history_length, command, gamepad, ghost)
      print(f"[INFO] Loaded velocity actor: {actor_path}", flush=True)
      print(
        f"[INFO] Gamepad: {gamepad.device} | Space: pause | Enter: reset", flush=True
      )
      GamepadVelocityViewer(env, policy, command).run()
    finally:
      env.close()
