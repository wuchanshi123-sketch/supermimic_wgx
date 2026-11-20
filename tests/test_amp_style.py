"""Regression coverage for the least-squares AMP prior."""

import math

import torch

from supermimic.learners.amp_style import (
  AmpDiscriminator,
  AmpStyleCfg,
  AmpStyleLearner,
  amp_command_gate,
)


def test_amp_command_gate_ramps_with_planar_speed_and_half_yaw():
  commands = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [0.06, 0.08, 0.0],
      [0.0, 0.0, -0.2],
      [0.08, 0.0, 0.03],
      [0.4, 0.0, 0.2],
    ]
  )
  torch.testing.assert_close(
    amp_command_gate(commands, threshold=0.2), torch.tensor([0.0, 0.5, 0.5, 0.475, 1.0])
  )
  torch.testing.assert_close(amp_command_gate(commands, threshold=0.0), torch.ones(5))


def test_amp_normalizer_merges_reference_and_both_training_endpoints_in_place():
  cfg = AmpStyleCfg(hidden_dims=())
  discriminator = AmpDiscriminator(feature_dim=2, cfg=cfg)
  learner = AmpStyleLearner(discriminator, cfg, device="cpu")
  expert_frames = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 10
  expert = expert_frames.reshape(3, 2, 2)
  policy = (expert + torch.tensor([-0.5, 0.25])).requires_grad_()
  pointers = [buffer.data_ptr() for buffer in discriminator.buffers()]
  discriminator.fit_normalizer(expert_frames)

  linear = discriminator.network[0]
  with torch.no_grad():
    linear.weight.fill_(0.1)
    linear.bias.zero_()
  before = linear.weight.detach().clone()
  logs = learner.update(expert, policy)

  all_frames = torch.cat(
    (expert_frames, expert.flatten(0, 1), policy.detach().flatten(0, 1))
  )
  variance, mean = torch.var_mean(all_frames, dim=0, correction=0)
  torch.testing.assert_close(discriminator.feature_mean, mean)
  torch.testing.assert_close(discriminator.feature_var, variance)
  assert discriminator.feature_count.item() == len(all_frames)
  assert [buffer.data_ptr() for buffer in discriminator.buffers()] == pointers
  assert not torch.equal(linear.weight, before)
  assert all(value.isfinite().item() for value in logs.values())
  assert policy.grad is None
  assert all(
    not param.requires_grad and param.grad is None
    for param in discriminator.parameters()
  )

  normalized_expert = ((expert - mean) / (variance + 1e-4).sqrt()).flatten(-2)
  normalized_policy = ((policy.detach() - mean) / (variance + 1e-4).sqrt()).flatten(-2)
  expert_score = (normalized_expert @ before.T).squeeze(-1)
  policy_score = (normalized_policy @ before.T).squeeze(-1)
  expected_classification = 0.5 * (
    (expert_score - 1).square().mean() + (policy_score + 1).square().mean()
  )
  torch.testing.assert_close(logs["AMP/discriminator_loss"], expected_classification)
  torch.testing.assert_close(
    logs["AMP/gradient_penalty"], cfg.gradient_penalty * before.square().sum()
  )


def test_amp_style_reward_keeps_vibe_curve_and_low_score_gradient():
  discriminator = AmpDiscriminator(feature_dim=2, cfg=AmpStyleCfg(hidden_dims=()))
  scores = torch.tensor(
    [-2.0, -1.0, -0.50001, -0.5, -0.49999, 0.0, 1.0, 2.0, 3.0, 4.0],
    requires_grad=True,
  )
  rewards = discriminator.reward_from_score(scores)
  torch.testing.assert_close(
    rewards[:2], torch.tensor([0.4375 * math.exp(-18 / 7), 0.4375 * math.exp(-6 / 7)])
  )
  torch.testing.assert_close(
    rewards[3::], torch.tensor([0.4375, 0.4375075, 0.75, 1.0, 0.75, 0.0, 0.0])
  )
  (slopes,) = torch.autograd.grad(rewards.sum(), scores)
  assert (slopes[:2] > 0).all()
  torch.testing.assert_close(slopes[2:5], torch.full((3,), 0.75), atol=2e-5, rtol=0)


def test_amp_style_reward_backpropagates_to_both_frames():
  discriminator = AmpDiscriminator(feature_dim=2, cfg=AmpStyleCfg(hidden_dims=()))
  with torch.no_grad():
    discriminator.network[0].weight.fill_(0.1)
    discriminator.network[0].bias.zero_()
  discriminator.requires_grad_(False)
  frames = torch.zeros(3, 2, 2, requires_grad=True)
  reward = discriminator.reward(frames[:, 0], frames[:, 1])
  torch.testing.assert_close(reward, torch.full((3,), 0.75))
  reward.sum().backward()

  assert frames.grad is not None
  torch.testing.assert_close(
    frames.grad, torch.full_like(frames, 0.05 / (1.0 + 1e-4) ** 0.5)
  )
  assert all(param.grad is None for param in discriminator.parameters())
