import hydra
import torch
from omegaconf import DictConfig

from optflow.diffusion.model import LatentTransformer
from optflow.diffusion.noise_scheduler import NoiseScheduler, ScheduleType
from optflow.dora import DoraVAE


@hydra.main(version_base=None, config_path="conf", config_name="config")
def test_diffusion(cfg: DictConfig) -> None:
    model = LatentTransformer(
        in_channels=cfg.diffusion.model.in_channels,
        width=cfg.diffusion.model.width,
        out_channels=cfg.diffusion.model.in_channels,  # out_channels should match in_channels
        num_heads=cfg.diffusion.model.num_heads,
        depth=cfg.diffusion.model.depth,
        num_freqs=cfg.diffusion.model.num_freqs,
        include_pi=cfg.diffusion.model.include_pi,
    ).to(cfg.device)

    noise_scheduler = NoiseScheduler(
        num_timesteps=cfg.diffusion.noise_scheduler.num_timesteps,
        beta_start=cfg.diffusion.noise_scheduler.beta_start,
        beta_end=cfg.diffusion.noise_scheduler.beta_end,
        beta_schedule=ScheduleType(cfg.diffusion.noise_scheduler.schedule_type),
        device=cfg.device,
    )

    x = torch.randn(
        cfg.batch_size, cfg.sequence_length, cfg.diffusion.model.in_channels
    ).to(cfg.device)
    noise = torch.randn_like(x).to(cfg.device)

    t = torch.randint(
        0, noise_scheduler.num_timesteps, (cfg.batch_size,), device=cfg.device
    )
    x_corrupted = noise_scheduler.add_noise(x, noise, t)

    y = model(x_corrupted, t.unsqueeze(-1))
    print(y.shape)
    assert y.shape == (
        cfg.batch_size,
        cfg.sequence_length,
        cfg.diffusion.model.in_channels,
    ), "Output shape mismatch"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def test_dora_pretrained_encoder(cfg: DictConfig) -> None:
    model = DoraVAE.from_pretrained_checkpoint(checkpoint_path=cfg.vae.checkpoint_path)
    model.to(cfg.device)

    model.encoder_mode()
    coarse_pc = torch.randn(cfg.batch_size, 16384, 3, device=cfg.device)
    coarse_feats = torch.randn(cfg.batch_size, 16384, 3, device=cfg.device)
    sharp_pc = torch.randn(cfg.batch_size, 2048, 3, device=cfg.device)
    sharp_feats = torch.randn(cfg.batch_size, 2048, 3, device=cfg.device)

    with torch.inference_mode():
        moments = model(
            coarse_pc=coarse_pc,
            coarse_feats=coarse_feats,
            sharp_pc=sharp_pc,
            sharp_feats=sharp_feats,
        )

    print("Moments shape:", moments.shape)
    assert moments.shape == (
        cfg.batch_size,
        cfg.vae.latent_sequence_length,
        cfg.vae.embed_dim * 2,
    ), "Moments shape mismatch"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def test_dora_pretrained_decoder(cfg: DictConfig) -> None:
    model = DoraVAE.from_pretrained_checkpoint(checkpoint_path=cfg.vae.checkpoint_path)
    model.to(cfg.device)

    model.decoder_mode()

    query_points = torch.randn(cfg.batch_size, 16384, 3, device=cfg.device)
    latents = torch.randn(
        cfg.batch_size,
        cfg.vae.latent_sequence_length,
        cfg.vae.embed_dim,
        device=cfg.device,
    )

    with torch.inference_mode():
        signed_distance = model(
            latents=latents,
            query_points=query_points,
        )

    print("SDF shape:", signed_distance.shape)
    assert signed_distance.shape == (
        cfg.batch_size,
        16384,
        cfg.vae.out_dims,
    ), "Signed distance shape mismatch"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def test_dora_pretrained_default(cfg: DictConfig) -> None:
    model = DoraVAE.from_pretrained_checkpoint(checkpoint_path=cfg.vae.checkpoint_path)
    model.to(cfg.device)

    model.default_mode()

    coarse_pc = torch.randn(cfg.batch_size, 16384, 3, device=cfg.device)
    coarse_feats = torch.randn(cfg.batch_size, 16384, 3, device=cfg.device)
    sharp_pc = torch.randn(cfg.batch_size, 2048, 3, device=cfg.device)
    sharp_feats = torch.randn(cfg.batch_size, 2048, 3, device=cfg.device)
    query_points = torch.randn(cfg.batch_size, 16384, 3, device=cfg.device)

    with torch.inference_mode():
        signed_distance = model(
            coarse_pc=coarse_pc,
            coarse_feats=coarse_feats,
            sharp_pc=sharp_pc,
            sharp_feats=sharp_feats,
            query_points=query_points,
        )

    print("SDF shape:", signed_distance.shape)
    assert signed_distance.shape == (
        cfg.batch_size,
        16384,
        cfg.vae.out_dims,
    ), "Signed distance shape mismatch"


if __name__ == "__main__":
    test_diffusion()
