import logging
import hydra
import numpy as np
import pyvista as pv
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from optflow.diffusion.model import LatentDDPM
from optflow.diffusion.noise_scheduler import NoiseScheduler, ScheduleType
from optflow.dora import DoraVAE
from skimage.measure import marching_cubes

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="sampling_config")
def sample_from_pretrained_diffusion_model(cfg: DictConfig) -> None:
    model = LatentDDPM.from_pretrained_checkpoint(
        checkpoint_path=cfg.diffusion.checkpoint_path, **cfg.diffusion.model
    ).to(cfg.device)
    noise_scheduler = NoiseScheduler(
        num_timesteps=cfg.diffusion.noise_scheduler.num_timesteps,
        beta_start=cfg.diffusion.noise_scheduler.beta_start,
        beta_end=cfg.diffusion.noise_scheduler.beta_end,
        schedule_type=ScheduleType(cfg.diffusion.noise_scheduler.schedule_type),
        device=cfg.device,
    )

    latent_samples = model.generate(
        num_samples=cfg.num_samples,
        latent_shape=(cfg.sequence_length, cfg.diffusion.model.in_channels),
        noise_scheduler=noise_scheduler,
        batch_size=cfg.batch_size,
        device=cfg.device,
    )

    if cfg.mesh_reconstruction.reconstruct_mesh:
        assert (
            cfg.num_samples % 4 == 0
        ), "I am enforcing that each row of the plot has four meshes, so ensure that num_samples is a multiple of four."
        pv.start_xvfb()
        plotter = pv.Plotter(
            shape=(cfg.num_samples // 4, 4), notebook=False, off_screen=True
        )
        plotter.set_background("lightgray")

        vae = DoraVAE.from_pretrained_checkpoint(
            checkpoint_path=cfg.mesh_reconstruction.checkpoint_path
        )
        vae.to(cfg.device)
        vae.eval()
        vae.decoder_mode()

        query_points = np.meshgrid(
            np.linspace(
                -1, 1, cfg.mesh_reconstruction.grid_size
            ),  # The assumption is that the mesh should lie in the [-1, 1] cube due to bounding_box centering and scaling
            np.linspace(-1, 1, cfg.mesh_reconstruction.grid_size),
            np.linspace(-1, 1, cfg.mesh_reconstruction.grid_size),
        )

        query_points = np.stack(query_points, axis=-1).reshape(-1, 3)
        query_points = torch.from_numpy(query_points).to(
            dtype=torch.float32, device=cfg.device
        )

        for idx, latent in tqdm(
            enumerate(latent_samples), desc="Reconstructing sampled latents"
        ):
            with torch.inference_mode():
                sdf = vae(
                    query_points=query_points.unsqueeze(0), latents=latent.unsqueeze(0)
                )
                sdf = sdf.detach().cpu().numpy().squeeze()
                sdf = sdf.reshape(
                    (
                        cfg.mesh_reconstruction.grid_size,
                        cfg.mesh_reconstruction.grid_size,
                        cfg.mesh_reconstruction.grid_size,
                    )
                )

            try:
                verts, faces, _, _ = marching_cubes(sdf, level=0.0)
                mesh = pv.PolyData.from_regular_faces(verts, faces)
                i = idx // 4
                j = idx % 4
                plotter.subplot(i, j)
                plotter.add_mesh(mesh, color="white")
            except ValueError as e:
                LOGGER.warning(f"Failed to reconstruct mesh: {e}")
                pass

        plotter.link_views()
        plotter.export_html("outputs/latent_sample_reconstructions.html")

        LOGGER.info(
            "Saved reconstruction to outputs/latent_sample_reconstructions.html"
        )
        plotter.close()


if __name__ == "__main__":
    sample_from_pretrained_diffusion_model()
