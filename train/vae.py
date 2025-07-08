import logging
from collections import deque
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange

import hydra
import torch
from schedulefree.adamw_schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import mlflow


from optflow.dora.dataset.dataset import DoraDataset
from optflow.dora.model import DoraVAE, InferenceMode
from optflow.utils.h5_dataset import H5Dataset

LOGGER = logging.getLogger(__name__)


def create_dora_dataloader(
    data_path: Path, mode: InferenceMode, cfg: DictConfig, shuffle: bool
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


def get_sdf_and_entropy_from_batch(
    batch: dict[str, torch.Tensor], vae: DoraVAE, device: str
) -> torch.Tensor:
    coarse_pc = batch["coarse_pc"].to(device)
    coarse_feats = batch["coarse_feats"].to(device)
    sharp_pc = batch["sharp_pc"].to(device)
    sharp_feats = batch["sharp_feats"].to(device)
    query_points = batch["query_points"].to(device)

    return vae(
        coarse_pc=coarse_pc,
        coarse_feats=coarse_feats,
        sharp_pc=sharp_pc,
        sharp_feats=sharp_feats,
        query_points=query_points,
        sample_posterior=True,
    )


def train(cfg, train_dataloader, val_dataloader, model, optimizer, loss_fn):
    training_losses = deque(maxlen=100)
    validation_losses = deque(maxlen=100)
    best_loss = np.inf
    with trange(
        cfg.n_epochs, desc="Training", unit="epoch", dynamic_ncols=True
    ) as pbar:
        for epoch in pbar:
            model.train()
            optimizer.train()
            with tqdm(
                train_dataloader, colour="#B5F2A9", unit="batch", dynamic_ncols=True
            ) as train_bar:
                batch_losses = deque(maxlen=100)
                for batch in train_bar:
                    sdf, entropy = get_sdf_and_entropy_from_batch(
                        batch, model, cfg.device
                    )

                    sdf_loss = loss_fn(
                        input=sdf,
                        target=batch["sdf"].unsqueeze(-1).to(cfg.device),
                        var=model.var.exp().expand(*sdf.shape),
                    )
                    entropy_loss = entropy
                    loss = sdf_loss + entropy_loss
                    if torch.any(loss.isnan()):
                        LOGGER.info("NaN encountered in loss, breaking.")
                        breakpoint()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    training_losses.append(loss.item())
                    batch_losses.append(loss.item())
                    train_bar.set_postfix(
                        {
                            "batch_loss": f"{loss.item():.4f}",
                        }
                    )
            mlflow.log_metric("train_loss", loss.item())
            mlflow.log_metric("train_sdf_nll", sdf_loss.item())
            mlflow.log_metric("train_kl_div", entropy_loss.item())
            model.eval()
            optimizer.eval()
            with tqdm(
                val_dataloader, colour="#F2A9B5", unit="batch", dynamic_ncols=True
            ) as val_bar:
                batch_losses = deque(maxlen=100)
                for batch in val_bar:
                    sdf, entropy = get_sdf_and_entropy_from_batch(
                        batch, model, cfg.device
                    )
                    sdf_loss = loss_fn(
                        input=sdf,
                        target=batch["sdf"].to(cfg.device),
                        var=model.var.exp().expand(*sdf.shape),
                    )
                    entropy_loss = entropy
                    loss = sdf_loss + entropy_loss

                    batch_losses.append(loss.item())
                    validation_losses.append(loss.item())
                    val_bar.set_postfix(
                        {
                            "batch_loss": f"{loss.item():.4f}",
                        }
                    )

            mlflow.log_metric("val_loss", loss.item())
            mlflow.log_metric("val_sdf_nll", sdf_loss.item())
            mlflow.log_metric("val_kl_div", entropy_loss.item())
            pbar.set_postfix(
                {
                    "epoch": epoch + 1,
                    "train_loss": f"{np.mean(training_losses):.4f}",
                    "val_loss": f"{np.mean(validation_losses):.4f}",
                }
            )

    if np.mean(validation_losses) < best_loss:
        print(
            f"Epoch {epoch}: Best loss improved {best_loss} -> {np.mean(validation_losses)}. Saving model."
        )
        torch.save(model.state_dict(), "outputs/diff_model_state_dict.pth")

    return model


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_dora_model(cfg: DictConfig):
    model = DoraVAE(**{k: v for k, v in cfg.vae.items() if k != "checkpoint_path"}).to(
        cfg.device
    )

    optimizer = AdamWScheduleFree(
        model.parameters(), cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )

    LOGGER.info(
        f"VAE model has {sum(p.numel() for p in model.parameters())} parameters."
    )

    data_base_path = Path(f"data/{cfg.dataset.name}")
    if not data_base_path.exists() and not data_base_path.is_dir():
        raise FileNotFoundError(
            f"Dataset path {data_base_path} does not exist or is not a directory."
        )
    train_data_path = data_base_path / "train"
    val_data_path = data_base_path / "test"

    train_dataloader = create_dora_dataloader(
        data_path=train_data_path, mode=model.mode, cfg=cfg, shuffle=True
    )
    val_dataloader = create_dora_dataloader(
        data_path=val_data_path, mode=model.mode, cfg=cfg, shuffle=False
    )

    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        mlflow.log_params(cfg)
        model = train(
            cfg,
            train_dataloader,
            val_dataloader,
            model,
            optimizer,
            loss_fn=torch.nn.GaussianNLLLoss(full=True, reduction="sum"),
        )

    return model


if __name__ == "__main__":
    model = train_dora_model()
    print("Finished training. Saving final model")
    torch.save(model.state_dict(), "outputs/final_dora_model_state_dict.pth")
