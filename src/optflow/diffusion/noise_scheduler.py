"""Noise Scheduler for Diffusion Models. Implementation reproduced from Baldassari et al. (2023)."""

from enum import Enum

import torch
from torch import nn


class ScheduleType(Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"


class NoiseScheduler:
    def __init__(
        self,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        schedule_type: ScheduleType,
        device: str,
    ):
        self.num_timesteps = num_timesteps
        match schedule_type:
            case ScheduleType.LINEAR:
                self.betas = torch.linspace(
                    beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device
                )
            # TODO: Quadratic schedule is not implemented yet.
            case _:
                raise ValueError(f"Unknown beta schedule: {schedule_type}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumulative = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumulative_prev = nn.functional.pad(
            self.alphas_cumulative[:-1], (1, 0), value=1.0
        )

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumulative_prev) / (1.0 - self.alphas_cumulative)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumulative_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumulative)
        )

    @property
    def sqrt_alphas_cumulative(self):
        return torch.sqrt(self.alphas_cumulative)

    @property
    def sqrt_one_minus_alphas_cumulative(self):
        return torch.sqrt(1.0 - self.alphas_cumulative)

    @property
    def sqrt_inverse_alphas_cumulative(self):
        return torch.sqrt(1.0 / self.alphas_cumulative)

    @property
    def sqrt_inverse_alphas_cumulative_minus_one(self):
        return torch.sqrt(1.0 / self.alphas_cumulative - 1.0)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inverse_alphas_cumulative[t]
        s2 = self.sqrt_inverse_alphas_cumulative_minus_one[t]
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t].repeat(1, 1, 1)
        s2 = self.posterior_mean_coef2[t].repeat(1, 1, 1)
        return s1 * x_0 + s2 * x_t

    def get_variance(self, t):
        if t == 0:
            return 0
        variance = (
            self.betas[t]
            * (1.0 - self.alphas_cumulative_prev[t])
            / (1.0 - self.alphas_cumulative[t])
        )
        return variance.clamp(min=1e-20)

    def step(self, model_output, t, sample):
        predicted_sample = self.reconstruct_x0(sample, t, model_output)
        predicted_prev_sample = self.q_posterior(predicted_sample, sample, t)
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = self.get_variance(t) ** 0.5 * noise

        return predicted_prev_sample + variance

    def add_noise(self, x0, noise, timesteps):
        s1 = self.sqrt_alphas_cumulative[timesteps][:, None, None]
        s2 = self.sqrt_one_minus_alphas_cumulative[timesteps][:, None, None]
        return s1 * x0 + s2 * noise

    def __len__(self):
        return self.num_timesteps
