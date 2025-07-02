from collections import deque
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange

import hydra
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from optflow.diffusion.model import LatentTransformer
from optflow.diffusion.noise_scheduler import NoiseScheduler, ScheduleType
from optflow.dora.dataset.dataset import DoraDataset
from optflow.dora.model import DoraVAE, InferenceMode
from optflow.utils.h5_dataset import H5Dataset


def create_dataloader(
    data_path: Path,
    mode: InferenceMode,
    cfg: DictConfig,
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
        shuffle=False,
        pin_memory=True,
    )

def get_latent_from_batch(batch: dict[str, torch.Tensor], vae: DoraVAE, device: str) -> torch.Tensor:
        coarse_pc = batch["coarse_pc"].to(device)
        coarse_feats = batch["coarse_feats"].to(device)
        sharp_pc = batch["sharp_pc"].to(device)
        sharp_feats = batch["sharp_feats"].to(device)

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

def setup_training_artifacts(cfg: DictConfig) -> LatentTransformer:
    model = LatentTransformer(
        in_channels=cfg.diffusion.model.in_channels,
        inner_product_channels=cfg.diffusion.model.inner_product_channels,
        out_channels=cfg.diffusion.model.in_channels,  # out_channels should match in_channels
        num_heads=cfg.diffusion.model.num_heads,
        depth=cfg.diffusion.model.depth,
        num_freqs=cfg.diffusion.model.num_freqs,
        include_pi=cfg.diffusion.model.include_pi,
    ).to(cfg.device)

    scheduler = NoiseScheduler(
        num_timesteps=cfg.diffusion.noise_scheduler.num_timesteps,
        beta_start=cfg.diffusion.noise_scheduler.beta_start,
        beta_end=cfg.diffusion.noise_scheduler.beta_end,
        beta_schedule=ScheduleType(cfg.diffusion.noise_scheduler.schedule_type),
        device=cfg.device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    return model, scheduler, optimizer



@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_diffusion_model(cfg: DictConfig):
    vae = DoraVAE.from_pretrained_checkpoint(checkpoint_path=cfg.vae.checkpoint_path)
    vae.to(cfg.device)
    vae.encoder_mode()
    vae.eval()

    model, noise_scheduler, optimizer = setup_training_artifacts(cfg)

    data_base_path = Path(f"data/{cfg.dataset.name}")
    if not data_base_path.exists() and not data_base_path.is_dir():
        raise FileNotFoundError(
            f"Dataset path {data_base_path} does not exist or is not a directory."
        )
    train_data_path = data_base_path / "train"
    train_dataloader = create_dataloader(
        data_path=train_data_path,
        mode=vae.mode,
        cfg=cfg
    )
    val_data_path = data_base_path / "test"
    val_dataloader = create_dataloader(
        data_path=val_data_path,
        mode=vae.mode,
        cfg=cfg
    )

    training_losses = deque(maxlen=100)
    validation_losses = deque(maxlen=100)
    best_loss = np.inf
    with trange(cfg.n_epochs, desc="Training", unit="epoch") as pbar:
        for epoch in pbar:
            model.train()
            with tqdm(train_dataloader , colour='#B5F2A9', unit='batch', dynamic_ncols=True) as train_bar:
                for batch in train_bar:
                    latents = get_latent_from_batch(batch, vae, cfg.device)
                    t = torch.randint(
                        0, noise_scheduler.num_timesteps, (cfg.batch_size,), device=latents.device
                    )
                    noise = torch.randn_like(latents)
                    latents_corrupted = noise_scheduler.add_noise(latents, noise, t)
                    pred_noise = model(latents_corrupted, t.unsqueeze(-1))
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(pred_noise, noise)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    training_losses.append(loss.item())
                    train_bar.set_postfix({
                        'running_batch_loss': f"{loss.item():.4f}",
                    })
            model.eval()
            with tqdm(val_dataloader, colour='#F2A9B5', unit='batch', dynamic_ncols=True) as val_bar:
                for batch in val_bar:
                    latents = get_latent_from_batch(batch, vae, cfg.device)
                    t = torch.randint(
                        0, noise_scheduler.num_timesteps, (cfg.batch_size,), device=latents.device
                    )
                    noise = torch.randn_like(latents)
                    latents_corrupted = noise_scheduler.add_noise(latents, noise, t)
                    pred_noise = model(latents_corrupted, t.unsqueeze(-1))
                    loss = torch.nn.functional.mse_loss(pred_noise, noise)

                    validation_losses.append(loss.item())
                    val_bar.set_postfix({
                        'batch_loss': f"{loss.item():.4f}",
                    })
            pbar.set_postfix({
                'epoch': epoch + 1,
                'train_loss': f"{np.mean(training_losses):.4f}",
                'val_loss': f"{np.mean(validation_losses):.4f}"
            })

    if np.mean(validation_losses) < best_loss:
        print(f"Epoch {epoch}: Best loss improved {best_loss} -> {np.mean(validation_losses)}. Saving model.")
        torch.save(model.state_dict(), "outputs/diff_model_state_dict.pth")



if __name__ == "__main__":
    model = train_diffusion_model()
    print("Finished training. Saving final model")
    torch.save(model.state_dict(), "outputs/final_diff_model_state_dict.pth")
