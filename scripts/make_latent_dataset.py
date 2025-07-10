import os
import h5py
import hydra
from pathlib import Path

from omegaconf import DictConfig
import torch
from tqdm import tqdm

from optflow.dora.model import DoraVAE
from scripts.generate_sdf_dora import create_dora_dataloader

_BASE_PATH = Path("/mnt/storage01/workspace/research/gen11/shapenet_13/car_dataset")
_LATENT_PATH = Path("data/car_latents_dataset")


def get_mean_logvar_from_batch(
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
        ).squeeze(0)
        mean, logvar = moments.chunk(2, dim=-1)
    return {
        "mean": mean.detach().cpu().numpy(),
        "logvar": logvar.detach().cpu().numpy(),
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def create_latent_dataset(cfg: DictConfig):
    vae = DoraVAE.from_pretrained_checkpoint(
        checkpoint_path=cfg.vae.checkpoint_path
    ).to(cfg.device)
    vae.encoder_mode()
    vae.eval()

    train_data_path = _BASE_PATH / "train"
    val_data_path = _BASE_PATH / "test"

    assert cfg.batch_size == 1
    train_dataloader = create_dora_dataloader(
        data_path=train_data_path, mode=vae.mode, cfg=cfg, shuffle=True
    )
    val_dataloader = create_dora_dataloader(
        data_path=val_data_path, mode=vae.mode, cfg=cfg, shuffle=False
    )

    for batch in tqdm(train_dataloader, desc="Generating train latents"):
        mean_logvar = get_mean_logvar_from_batch(batch, vae, cfg.device)
        output_file_path = _LATENT_PATH / "train" / f"{batch['name'][0]}.h5"

        with h5py.File(output_file_path, "w") as f:
            for key, value in mean_logvar.items():
                f.create_dataset(key, data=value)

    for batch in tqdm(val_dataloader, desc="Generating test latents"):
        mean_logvar = get_mean_logvar_from_batch(batch, vae, cfg.device)
        output_file_path = _LATENT_PATH / "test" / f"{batch['name'][0]}.h5"

        with h5py.File(output_file_path, "w") as f:
            for key, value in mean_logvar.items():
                f.create_dataset(key, data=value)

    return


if __name__ == "__main__":
    os.makedirs(_LATENT_PATH, exist_ok=True)
    os.makedirs(_LATENT_PATH / "train", exist_ok=True)
    os.makedirs(_LATENT_PATH / "test", exist_ok=True)

    create_latent_dataset()
