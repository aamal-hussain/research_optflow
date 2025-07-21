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
        eta: float = 0.0,
    ):
        self.num_timesteps = num_timesteps
        self.eta = eta

        match schedule_type:
            case ScheduleType.LINEAR:
                self.betas = torch.linspace(
                    beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device
                )
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

    def get_ddpm_variance(self, t):
        if t == 0:
            return 0
        variance = (
            self.betas[t]
            * (1.0 - self.alphas_cumulative_prev[t])
            / (1.0 - self.alphas_cumulative[t])
        )
        return variance.clamp(min=1e-20)

    def get_ddim_variance(self, t):
        if t == 0:
            return 0.0

        ddim_variance = (
            self.eta ** 2
            * self.betas[t]
            * (1.0 - self.alphas_cumulative_prev[t])
            / (1.0 - self.alphas_cumulative[t])
        )
        return ddim_variance.clamp(min=1e-20)

    def step(self, model_output, t, sample):
        if isinstance(t, int):
            t_idx = torch.tensor([t], device=sample.device)
        else:
            t_idx = t

        x0_hat = self.reconstruct_x0(sample, t_idx, model_output)

        alpha_cumulative_prev_t = self.alphas_cumulative_prev[t_idx]
        
        sqrt_alpha_cumulative_prev = torch.sqrt(alpha_cumulative_prev_t)
        
        ddim_variance_t = self.get_ddim_variance(t_idx)

        coeff_noise = torch.sqrt(torch.clamp(1.0 - alpha_cumulative_prev_t - ddim_variance_t, min=0.0))

        predicted_prev_sample_mean = sqrt_alpha_cumulative_prev * x0_hat + coeff_noise * model_output

        variance_noise = 0
        if self.eta > 0 and t_idx.item() > 0:
            z = torch.randn_like(model_output)
            variance_noise = torch.sqrt(ddim_variance_t) * z

        return predicted_prev_sample_mean + variance_noise

    def add_noise(self, x0, noise, timesteps):
        s1 = self.sqrt_alphas_cumulative[timesteps][:, None, None]
        s2 = self.sqrt_one_minus_alphas_cumulative[timesteps][:, None, None]
        return s1 * x0 + s2 * noise

    def __len__(self):
        return self.num_timesteps