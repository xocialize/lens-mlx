"""FlowMatchEulerDiscreteScheduler — minimal inference port for Lens.

Mirrors the diffusers scheduler as the Lens pipeline drives it: explicit `sigmas`
(= linspace(1, 1/N, N)) + an explicit precomputed `mu` (T6) with exponential dynamic
shift. `base_shift`/`max_shift` are unused because mu is precomputed.

Sanity vs golden: sigmas[0]=1.0 -> shift(mu,1.0)=exp(mu)/exp(mu)=1.0 -> timestep
1000 -> the DiT sees timestep/1000 = 1.0 (== golden dit_in_timestep).
"""

from __future__ import annotations

import math

import mlx.core as mx


class FlowMatchEulerDiscreteScheduler:
    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.sigmas = None
        self.timesteps = None
        self._step_index = 0

    @staticmethod
    def _time_shift_exponential(mu: float, sigma: float, t):
        # diffusers exponential time shift: exp(mu) / (exp(mu) + (1/t - 1)**sigma)
        return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)

    def set_timesteps(self, sigmas, mu: float):
        shifted = [self._time_shift_exponential(mu, 1.0, float(s)) for s in sigmas]
        self.timesteps = [s * self.num_train_timesteps for s in shifted]
        self.sigmas = shifted + [0.0]  # terminal 0
        self._step_index = 0

    def step(self, model_output: mx.array, sample: mx.array) -> mx.array:
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        prev = sample + (sigma_next - sigma) * model_output
        self._step_index += 1
        return prev
