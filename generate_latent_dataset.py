from pathlib import Path
import h5py
from tqdm import tqdm

import hydra
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from optflow.dora.dataset.dataset import DoraDataset
from optflow.dora.model import DoraVAE, InferenceMode
from optflow.utils.h5_dataset import H5Dataset


_SPLIT = "train"

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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def generate_latent_dataset(cfg: DictConfig):
    model = DoraVAE.from_pretrained_checkpoint(checkpoint_path=cfg.vae.checkpoint_path)
    model.to(cfg.device)
    model.encoder_mode()
    model.eval()


    data_base_path = Path(f"data/{cfg.dataset.name}")
    if not data_base_path.exists() and not data_base_path.is_dir():
        raise FileNotFoundError(
            f"Dataset path {data_base_path} does not exist or is not a directory."
        )
    data_path = data_base_path / _SPLIT
    dataloader = create_dataloader(
        data_path=data_path,
        mode=model.mode,
        cfg=cfg
    )

    samples = {}

    with torch.inference_mode():
        for i, batch in tqdm(enumerate(dataloader), desc="Generating Latent Dataset"):
            coarse_pc = batch["coarse_pc"].to(cfg.device)
            coarse_feats = batch["coarse_feats"].to(cfg.device)
            sharp_pc = batch["sharp_pc"].to(cfg.device)
            sharp_feats = batch["sharp_feats"].to(cfg.device)

            moments = model(
                coarse_pc=coarse_pc,
                coarse_feats=coarse_feats,
                sharp_pc=sharp_pc,
                sharp_feats=sharp_feats,
            )
            mean, logvar = moments.chunk(2, dim=-1)
            std = torch.exp(0.5 * logvar)
            latents = mean + std * torch.randn_like(mean)

            samples[str(i).zfill(5)] = {"latents": latents.detach().cpu().numpy().squeeze()}


    latents_data_path = data_base_path / "latents" / _SPLIT
    latents_data_path.mkdir(parents=True, exist_ok=True)
    for name, sample in tqdm(samples.items(), desc="Saving Latents"):
        with h5py.File(latents_data_path / f"{name}.h5", "w") as f:
            f.create_dataset("latents", data=sample["latents"])
        f.close()

    print(f"Latent dataset saved to {latents_data_path}")

if __name__ == "__main__":
    generate_latent_dataset()
