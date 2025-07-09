import logging
from collections import deque
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange

import hydra
import torch
from schedulefree.adamw_schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import mlflow

from optflow.diffusion.latent_dataset import LatentDataset
from optflow.diffusion.model import LatentDDPM
from optflow.diffusion.noise_scheduler import NoiseScheduler, ScheduleType
from optflow.dora.dataset.dataset import DoraDataset
from optflow.dora.model import DoraVAE, VAEMode
from optflow.utils.h5_dataset import H5Dataset

LOGGER = logging.getLogger(__name__)


def create_dora_dataloader(
    data_path: Path, mode: VAEMode, cfg: DictConfig, shuffle: bool
):
    if (
        not data_path.exists()
        and not data_path.is_dir()
        and not any(data_path.glob("*.h5"))
    ):
        raise FileNotFoundError(
            f"Dataset path {data_path} does not exist or is not a directory with .h5 files."
        )

    data = H5Dataset(data_path)
    dataset = DoraDataset(data=data, mode=mode, **cfg.dataset.params)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )


def create_latent_dataloader(data_path: Path, cfg: DictConfig, shuffle: bool):
    if (
        not data_path.exists()
        and not data_path.is_dir()
        and not any(data_path.glob("*.h5"))
    ):
        raise FileNotFoundError(
            f"Dataset path {data_path} does not exist or is not a directory with .h5 files."
        )

    data = H5Dataset(data_path)
    dataset = LatentDataset(data=data, **cfg.dataset.params)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )


def get_latent_from_batch(
    batch: dict[str, torch.Tensor], vae: DoraVAE, device: str
) -> torch.Tensor:
    coarse_pc = batch["coarse_pc"].to(device)
    coarse_feats = batch["coarse_feats"].to(device)
    sharp_pc = batch["sharp_pc"].to(device)
    sharp_feats = batch["sharp_feats"].to(device)

    with torch.inference_mode():
        moments = vae(
            coarse_pc=coarse_pc,
            coarse_feats=coarse_feats,
            sharp_pc=sharp_pc,
            sharp_feats=sharp_feats,
        )
        mean, logvar = moments.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        latents = mean + std * torch.randn_like(mean)
    return latents


def setup_training_artifacts(cfg: DictConfig) -> LatentDDPM:
    model = LatentDDPM(**cfg.diffusion.model).to(cfg.device)

    cfg.diffusion.noise_scheduler.schedule_type = ScheduleType(
        cfg.diffusion.noise_scheduler.schedule_type
    )
    scheduler = NoiseScheduler(device=cfg.device, **cfg.diffusion.noise_scheduler)

    optimizer = AdamWScheduleFree(model.parameters(), **cfg.optimizer)

    return model, scheduler, optimizer


def train(
    cfg,
    train_dataloader,
    val_dataloader,
    vae,
    model,
    noise_scheduler,
    optimizer,
    run_id,
):
    best_loss = np.inf
    train_batch_losses = deque(maxlen=len(train_dataloader))
    val_batch_losses = deque(maxlen=len(val_dataloader))
    with trange(
        cfg.n_epochs, desc="Training", unit="epoch", dynamic_ncols=True
    ) as pbar:
        for epoch in pbar:
            model.train()
            optimizer.train()
            with tqdm(
                train_dataloader, colour="#B5F2A9", unit="batch", dynamic_ncols=True
            ) as train_bar:
                for batch in train_bar:
                    latents = (
                        get_latent_from_batch(batch, vae, cfg.device)
                        if vae
                        else batch.to(cfg.device)
                    )
                    t = torch.randint(
                        0,
                        noise_scheduler.num_timesteps,
                        (latents.shape[0],),
                        device=latents.device,
                    )
                    noise = torch.randn_like(latents)
                    latents_corrupted = noise_scheduler.add_noise(latents, noise, t)
                    pred_noise = model(latents_corrupted, t.unsqueeze(-1))
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(pred_noise, noise)
                    if torch.any(loss.isnan()):
                        LOGGER.info("NaN encountered in loss, breaking.")
                        breakpoint()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_batch_losses.append(loss.item())
                    train_bar.set_postfix(
                        {
                            "batch_loss": f"{loss.item():.4f}",
                        }
                    )
            mlflow.log_metric("train_loss", np.mean(train_batch_losses), step=epoch)

            model.eval()
            optimizer.eval()
            with tqdm(
                val_dataloader, colour="#F2A9B5", unit="batch", dynamic_ncols=True
            ) as val_bar:
                for batch in val_bar:
                    latents = (
                        get_latent_from_batch(batch, vae, cfg.device)
                        if vae
                        else batch.to(cfg.device)
                    )
                    t = torch.randint(
                        0,
                        noise_scheduler.num_timesteps,
                        (latents.shape[0],),
                        device=latents.device,
                    )
                    noise = torch.randn_like(latents)
                    latents_corrupted = noise_scheduler.add_noise(latents, noise, t)
                    pred_noise = model(latents_corrupted, t.unsqueeze(-1))
                    loss = torch.nn.functional.mse_loss(pred_noise, noise)

                    val_batch_losses.append(loss.item())
                    val_bar.set_postfix(
                        {
                            "batch_loss": f"{loss.item():.4f}",
                        }
                    )

            mlflow.log_metric("val_loss", np.mean(val_batch_losses), step=epoch)
            pbar.set_postfix(
                {
                    "epoch": epoch + 1,
                    "train_loss": f"{np.mean(train_batch_losses):.4f}",
                    "validation_loss": f"{np.mean(val_batch_losses):.4f}",
                }
            )

            c_val_loss = np.mean(val_batch_losses)
            if c_val_loss < best_loss:
                print(
                    f"Epoch {epoch}: Best loss improved {best_loss:.4f} -> {c_val_loss:.4f}. Saving model."
                )
                best_loss = c_val_loss
                torch.save(model.state_dict(), f"outputs/{run_id}/model_state_dict.pth")

    print("Finished training. Saving final model")
    torch.save(model.state_dict(), f"outputs/{run_id}/final_model_state_dict.pth")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_diffusion_model(cfg: DictConfig):
    model, noise_scheduler, optimizer = setup_training_artifacts(cfg)
    LOGGER.info(
        f"Diffusion model has {sum(p.numel() for p in model.parameters())} parameters."
    )

    data_base_path = Path(f"data/{cfg.dataset.name}")
    if not data_base_path.exists() and not data_base_path.is_dir():
        raise FileNotFoundError(
            f"Dataset path {data_base_path} does not exist or is not a directory."
        )
    train_data_path = data_base_path / "train"
    val_data_path = data_base_path / "test"

    if "latents" not in cfg.dataset.name:
        vae = DoraVAE.from_pretrained_checkpoint(
            checkpoint_path=cfg.vae.checkpoint_path
        ).to(cfg.device)
        vae.encoder_mode()
        vae.eval()
        LOGGER.info(
            f"VAE model has {sum(p.numel() for p in vae.parameters())} parameters."
        )
        train_dataloader = create_dora_dataloader(
            data_path=train_data_path, mode=vae.mode, cfg=cfg, shuffle=True
        )
        val_dataloader = create_dora_dataloader(
            data_path=val_data_path, mode=vae.mode, cfg=cfg, shuffle=False
        )

    else:
        vae = None
        LOGGER.info("Not using a VAE model - using latents directly")
        train_dataloader = create_latent_dataloader(
            data_path=train_data_path, cfg=cfg, shuffle=True
        )
        val_dataloader = create_latent_dataloader(
            data_path=val_data_path, cfg=cfg, shuffle=False
        )

    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        run_id = mlflow.active_run().info.run_id
        os.makedirs(f"outputs/{run_id}")
        mlflow.log_params(cfg)
        train(
            cfg,
            train_dataloader,
            val_dataloader,
            vae,
            model,
            noise_scheduler,
            optimizer,
            run_id,
        )


if __name__ == "__main__":
    train_diffusion_model()
