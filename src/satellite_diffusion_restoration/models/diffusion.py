"""Small DDPM scheduler for conditional restoration experiments."""

from __future__ import annotations

import torch


class DDPMScheduler:
    """Linear-beta DDPM schedule with forward noising and reverse sampling helpers."""

    def __init__(
        self,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | str = "cpu",
    ) -> None:
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = torch.device(device)
        self._build_schedule()

    def to(self, device: torch.device | str) -> "DDPMScheduler":
        """Move schedule tensors to a device and return self."""
        self.device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(self.device))
        return self

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Diffuse clean image ``x0`` into ``x_t`` at the provided timesteps."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar_t = self.extract(self.sqrt_alpha_bar, timesteps, x0.shape)
        sqrt_one_minus_alpha_bar_t = self.extract(
            self.sqrt_one_minus_alpha_bar,
            timesteps,
            x0.shape,
        )
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate clean ``x0`` from ``x_t`` and predicted noise."""
        sqrt_alpha_bar_t = self.extract(self.sqrt_alpha_bar, timesteps, x_t.shape)
        sqrt_one_minus_alpha_bar_t = self.extract(
            self.sqrt_one_minus_alpha_bar,
            timesteps,
            x_t.shape,
        )
        return (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Sample ``x_{t-1}`` from ``x_t`` using the DDPM posterior mean."""
        predicted_x0 = self.predict_x0_from_noise(x_t, timesteps, predicted_noise).clamp(0.0, 1.0)
        coef1 = self.extract(self.posterior_mean_coef1, timesteps, x_t.shape)
        coef2 = self.extract(self.posterior_mean_coef2, timesteps, x_t.shape)
        mean = coef1 * predicted_x0 + coef2 * x_t

        posterior_variance_t = self.extract(self.posterior_variance, timesteps, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = (timesteps != 0).float().view(-1, 1, 1, 1)
        return mean + nonzero_mask * torch.sqrt(posterior_variance_t.clamp_min(1e-20)) * noise

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        corrupted: torch.Tensor,
        shape: tuple[int, int, int, int] | None = None,
        sampler: str = "ddpm",
        sample_steps: int | None = None,
    ) -> torch.Tensor:
        """Restore images by starting from noise and conditioning on corrupted inputs."""
        if sampler == "ddim":
            return self.sample_ddim(model, corrupted, shape=shape, sample_steps=sample_steps)
        if sampler != "ddpm":
            raise ValueError(f"Unknown sampler: {sampler!r}. Expected 'ddpm' or 'ddim'.")

        model_was_training = model.training
        model.eval()
        sample_shape = shape or corrupted.shape
        x_t = torch.randn(sample_shape, device=corrupted.device, dtype=corrupted.dtype)

        for step in reversed(range(self.timesteps)):
            timesteps = torch.full(
                (sample_shape[0],),
                step,
                device=corrupted.device,
                dtype=torch.long,
            )
            predicted_noise = model(x_t, corrupted, timesteps)
            x_t = self.p_sample(x_t, timesteps, predicted_noise)

        if model_was_training:
            model.train()
        return x_t.clamp(0.0, 1.0)

    @torch.no_grad()
    def sample_ddim(
        self,
        model: torch.nn.Module,
        corrupted: torch.Tensor,
        shape: tuple[int, int, int, int] | None = None,
        sample_steps: int | None = None,
    ) -> torch.Tensor:
        """Deterministic DDIM-style sampling with fewer reverse steps."""
        model_was_training = model.training
        model.eval()
        sample_shape = shape or corrupted.shape
        step_count = min(sample_steps or self.timesteps, self.timesteps)
        step_indices = torch.linspace(
            self.timesteps - 1,
            0,
            step_count,
            device=corrupted.device,
        ).round().long()
        step_indices = torch.unique_consecutive(step_indices)
        x_t = torch.randn(sample_shape, device=corrupted.device, dtype=corrupted.dtype)

        for index, step in enumerate(step_indices):
            timesteps = torch.full(
                (sample_shape[0],),
                int(step.item()),
                device=corrupted.device,
                dtype=torch.long,
            )
            predicted_noise = model(x_t, corrupted, timesteps)
            predicted_x0 = self.predict_x0_from_noise(
                x_t,
                timesteps,
                predicted_noise,
            ).clamp(0.0, 1.0)

            if index == len(step_indices) - 1:
                x_t = predicted_x0
                continue

            previous_step = step_indices[index + 1]
            alpha_bar_previous = self.alpha_bar[previous_step].view(1, 1, 1, 1)
            x_t = (
                torch.sqrt(alpha_bar_previous) * predicted_x0
                + torch.sqrt(1.0 - alpha_bar_previous) * predicted_noise
            )

        if model_was_training:
            model.train()
        return x_t.clamp(0.0, 1.0)

    def extract(
        self,
        values: torch.Tensor,
        timesteps: torch.Tensor,
        target_shape: torch.Size | tuple[int, ...],
    ) -> torch.Tensor:
        """Gather a 1D schedule tensor for a batch and reshape for broadcasting."""
        gathered = values.gather(0, timesteps.to(values.device))
        return gathered.reshape(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))

    def state_dict(self) -> dict[str, float | int]:
        """Return minimal serializable scheduler configuration."""
        return {
            "timesteps": self.timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
        }

    def _build_schedule(self) -> None:
        self.betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.timesteps,
            dtype=torch.float32,
            device=self.device,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        alpha_bar_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alpha_bar[:-1]],
            dim=0,
        )

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.posterior_variance = self.betas * (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(alpha_bar_prev) / (
            1.0 - self.alpha_bar
        )
        self.posterior_mean_coef2 = (
            (1.0 - alpha_bar_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bar)
        )
