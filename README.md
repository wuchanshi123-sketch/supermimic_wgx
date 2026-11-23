# SuperMimic

[Project page](https://supermimic.github.io) | [Paper](https://arxiv.org/abs/2610.28480)

SuperMimic trains a humanoid whole-body motion tracker with one differentiable world model, built on [mjlab](https://github.com/mujocolab/mjlab).

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone git@github.com:HybridRobotics/supermimic.git
git clone git@github.com:HybridRobotics/mjlab_research.git
cd supermimic
```

## Usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

Training saves the initial weights in `checkpoint_0/` before the first
iteration. By default, it then saves a separate checkpoint every 2,000 completed
iterations (`checkpoint_2000/`, `checkpoint_4000/`, and so on). Each checkpoint
is under `logs/<TASK_ID>/<run-timestamp>/checkpoint_<iteration>/`.
The run timestamp uses `YYYY-MM-DD_HH-MM-SS`.
The run directory links `actor.pt`, `world_model.pt`, `critic.pt` (when saved),
and `runner_config.json` to the latest checkpoint.

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

To play a local SuperMimic checkpoint in the native MuJoCo window without W&B,
run this from a terminal in your desktop session:

```bash
uv run src/supermimic/scripts/play.py <TASK_ID> \
  --checkpoint "logs/<TASK_ID>/<run-timestamp>/actor.pt" \
  --motion-file path/to/motion.npz \
  --device cpu --viewer native
```

To play a particular iteration, point `--checkpoint` to its `actor.pt`, for
example `checkpoint_2000/actor.pt`. Keep the matching `world_model.pt` next to
`actor.pt`. Playback reads `runner_config.json` from the same directory when
available. For an older checkpoint, playback uses the task's default runner
settings unless you pass `--runner-config` with the saved `{"runner_cfg": {...}}`
JSON.

Press **Space** to pause or resume, **Enter** to reset, and **Right Arrow** to
advance one step while paused. Omit `--device cpu` to use CUDA when available.
Use `--viewer viser` to play in a browser.

### G1 velocity following

`SuperMimic-GenCoord-G1-Velocity` learns to follow sampled forward, lateral, and
yaw-rate commands on flat ground. It uses the differentiable world model and
its actor-critic learner. Training starts from the G1 standing pose and
collects its own MuJoCo data.

```bash
CUDA_VISIBLE_DEVICES=0 uv run src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-G1-Velocity --num-envs 2048 \
  --runner.run-name supermimic_g1_velocity --runner.max-iterations 100000 \
  --env.sim.mujoco.iterations 50 --env.sim.mujoco.ls-iterations 50
```

The default limits are ±1.0 m/s forward and backward, ±1.0 m/s sideways, and
±3.0 rad/s in yaw. Set the ranges with
`--env.commands.twist.lin-vel-x`, `--env.commands.twist.lin-vel-y`, and
`--env.commands.twist.ang-vel-z`; pass each range as a quoted tuple, such as
`'(-1.0,1.0)'`. `Episode_Metrics/vx_error`, `vy_error`, and `wz_error` report
measured tracking errors in MuJoCo.

The velocity task uses the non-foot rewards from Unitree G1 Flat, plus its
foot slip (`-0.25`) and soft landing (`-0.001`) costs. Slip penalizes the
squared horizontal velocity of contacting feet. Soft landing penalizes force
magnitude on first contact. Both apply when the command's planar speed plus
absolute yaw rate exceeds `0.1`. Speed commands and episode termination
conditions follow the SuperMimic velocity task.

MuJoCo and the imagined rollouts share the reward formula. MuJoCo measures
joint acceleration, self-collision count, foot slip and first-contact force
directly. During training, the world model fits an auxiliary predictor for
those signals from the collected measurements. Its weights are saved in
`world_model.pt`. Older checkpoints restore any existing prediction weights
and learn the newly added outputs during warm-up.

To start a new run from an existing velocity policy, pass
`--runner.init-checkpoint` with a checkpoint directory containing `actor.pt`
and `world_model.pt`. The new run uses the saved actor during replay warm-up
and trains a fresh critic with the current rewards. `--runner.max-iterations`
counts iterations in the new run, which starts saving at `checkpoint_0`.

### G1 velocity following with AMP

`SuperMimic-GenCoord-G1-Velocity-AMP` adds a learned gait-style reward to the
velocity task. Following `supermimic_vibe`, the discriminator looks at two
adjacent frames of 13 G1 links. Each frame contains link poses relative to the
torso and linear and angular velocities in the links' local frames. The
discriminator learns from real MuJoCo replay and reference clips. The actor
receives the style reward during differentiable world-model rollouts. This task
uses the velocity reward without the whole-body angular momentum penalty.

```bash
CUDA_VISIBLE_DEVICES=0 uv run src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-G1-Velocity-AMP --num-envs 2048 \
  --runner.amp.motion-file ../WalkandRun_amp_walk_50hz \
  --runner.run-name supermimic_g1_velocity_amp --runner.max-iterations 100000 \
  --env.sim.mujoco.iterations 50 --env.sim.mujoco.ls-iterations 50
```

The AMP task defaults to the sibling `WalkandRun_amp_walk_50hz` directory.
`--runner.amp.motion-file` also accepts a single `.npz` or a quoted glob. The
file frame rate must match the controller rate. Clips are sampled equally,
then an adjacent pair is sampled within each selected clip. The discriminator
updates running normalization from both reference and replay frames and uses
two least-squares batches per world-model update, with a gradient penalty on
reference samples.

`--runner.amp.weight` defaults to `1.25`, which gives a maximum style bonus
of `0.025` per step at 50 Hz. The reward follows Vibe's curve above a
discriminator score of `-0.5` and uses a smooth exponential tail below that
point so that poorly scored policy transitions can still receive a style
gradient. The bonus ramps up from zero with the command magnitude. Speed
tracking is rewarded separately and does not reduce the style bonus by default.
Use `--runner.amp.style-speed-gate True` to restore the linear and yaw tracking
error multiplier for comparison.
`AMP/style_reward` reports the mean unweighted style score on real replay,
before command and tracking factors. `AMP/expert_score`, `AMP/policy_score`
and `AMP/gradient_penalty` describe discriminator training.

`world_model.pt` stores the discriminator and normalization statistics.
`--runner.init-checkpoint` restores the actor and dynamics. It restores AMP
weights from checkpoints using the same feature and objective version. Older
AMP checkpoints initialize a fresh discriminator for the link features.
AMP warm starts cap the actor learning rate at `1e-5` and penalize deviations
from the saved actor's actions on the same imagined observations. The critic
uses its normal learning rate. The cap and action penalty can be adjusted with
`--runner.amp.warm-start-actor-lr` and `--runner.amp.action-anchor-weight`.
Local playback uses the trained actor, plus the world model for the
prediction ghost.

### G1 velocity playback

Drive a local velocity actor with a Linux gamepad in the native MuJoCo window.

```bash
uv run src/supermimic/scripts/play.py SuperMimic-GenCoord-G1-Velocity \
  --checkpoint "logs/SuperMimic-GenCoord-G1-Velocity/<run-timestamp>/actor.pt" \
  --gamepad-device /dev/input/js1
```

The left stick controls forward/backward and lateral velocity (both ±1.0 m/s).
The right stick controls yaw rate (±3.0 rad/s). The top-right overlay shows
the `vx`, `vy`, and `wz` targets sent to the policy after deadzone and speed
scaling. Centered sticks and a disconnected gamepad give a zero command.
**Space** pauses or resumes, and **Enter** resets the robot. Use
`ls -l /dev/input/by-id/*-joystick` to locate a stable device path. The player
reads `actor.pt` directly and uses the sibling `runner_config.json` to recover
a custom action limit when available.

When `world_model.pt` is present in the same checkpoint, the window also
shows the world model prediction as a translucent blue robot. Both robots
receive the same gamepad command. The ghost starts from the simulator state
and follows its own predicted state for 100 control steps (2 seconds) before
resynchronizing. Set `--num-steps-rollout` to change that interval. **R** toggles
the ghost with the other debug visuals. Press **Enter** to reset both robots.

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
