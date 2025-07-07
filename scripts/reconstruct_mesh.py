from pathlib import Path
import h5py
import numpy as np
import pyvista as pv

import hydra
import torch

from omegaconf import DictConfig

from optflow.dora.dataset.dataset import DoraDataset
from optflow.dora.model import DoraVAE
from skimage.measure import marching_cubes



def reconstruct_mesh(sample, model, cfg, name):
    verts = sample[cfg.dataset_params.verts_key]
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    query_points = np.meshgrid(
        np.linspace(bbox_min[0], bbox_max[0], cfg.grid_size),
        np.linspace(bbox_min[1], bbox_max[1], cfg.grid_size),
        np.linspace(bbox_min[2], bbox_max[2], cfg.grid_size),
    )
    query_points = np.stack(query_points, axis=-1).reshape(-1, 3)

    dataset = DoraDataset([sample], mode=model.mode, **cfg.dataset_params)
    model_input = dataset[0]

    coarse_pc = model_input["coarse_pc"].to(cfg.device).unsqueeze(0)
    coarse_feats = model_input["coarse_feats"].to(cfg.device).unsqueeze(0)
    sharp_pc = model_input["sharp_pc"].to(cfg.device).unsqueeze(0)
    sharp_feats = model_input["sharp_feats"].to(cfg.device).unsqueeze(0)

    query_points = torch.from_numpy(query_points).to(device=cfg.device, dtype=torch.float32).unsqueeze(0)
    sdf = model(
        coarse_pc=coarse_pc,
        coarse_feats=coarse_feats,
        sharp_pc=sharp_pc,
        sharp_feats=sharp_feats,
        query_points=query_points,
        sample_posterior=cfg.sample_posterior,
    ).squeeze(-1)


    sdf = sdf.detach().cpu().numpy().squeeze()
    sdf = sdf.reshape((cfg.grid_size, cfg.grid_size, cfg.grid_size))

    verts, faces, _, _ = marching_cubes(sdf, level=0.0)

    mesh = pv.PolyData.from_regular_faces(verts, faces)
    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color="white")
    plotter.export_html(f"outputs/{name}.html")
    plotter.close()

@hydra.main(version_base=None, config_path="../conf", config_name="reconstruction_config")
def reconstruct_data_sample(cfg: DictConfig):
    model = DoraVAE.from_pretrained_checkpoint(checkpoint_path=cfg.checkpoint_path)
    model.to(cfg.device)
    model.eval()

    dataset_path = Path(f"data/{cfg.dataset_name}")
    data_sample = list(dataset_path.rglob(f"{cfg.sample_name}.h5"))
    assert len(data_sample) > 0, f"Sample {cfg.sample_name} not found in {cfg.dataset_name}"
    data_sample = data_sample[0]

    sample = {}
    with h5py.File(data_sample, 'r') as f:
        for key in f.keys():
            sample[key] = f[key][()]
        f.close()

    reconstruct_mesh(sample, model, cfg, cfg.sample_name)


@hydra.main(version_base=None, config_path="../conf", config_name="reconstruction_config")
def reconstruct_pyvista_primitive(cfg: DictConfig):
    model = DoraVAE.from_pretrained_checkpoint(checkpoint_path=cfg.checkpoint_path)
    model.eval()
    model.to(cfg.device)


    match cfg.pyvista_primitive:
        case "bunny":
            mesh = pv.examples.download_bunny_coarse()
        case "airplane":
            mesh = pv.examples.load_airplane()
        case "armadillo":
            mesh = pv.examples.download_armadillo()
        case _:
            raise ValueError(f"{cfg.pyvista_primitive} is not available.")

    mesh = mesh.clean(absolute=False, tolerance=1e-8)
    mesh = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False)
    mesh = center_and_scale_mesh(mesh)

    sample = {
            cfg.dataset_params.verts_key: mesh.points,
            cfg.dataset_params.faces_key: mesh.regular_faces,
            cfg.dataset_params.areas_key: mesh.cell_data["Area"],
            cfg.dataset_params.faces_normals_key: mesh.cell_normals,
    }

    reconstruct_mesh(sample, model, cfg, cfg.pyvista_primitive)

def center_and_scale_mesh(mesh: pv.PolyData):
    mesh = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    mesh.cell_data["Area"] = np.maximum(0, mesh.cell_data["Area"])
    center, scale = calculate_center_and_scale_of_mesh(
        verts=mesh.points,
    )
    mesh.points = ((mesh.points - center) / scale).astype(mesh.points.dtype)
    mesh.cell_data["Area"] /= scale * scale
    return mesh

def calculate_center_and_scale_of_mesh(
    verts: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Normalize mesh vertices to [-1, 1]."""
    bounding_box_min = verts.min(axis=0)
    bounding_box_max = verts.max(axis=0)
    center = (bounding_box_max + bounding_box_min) / 2.0
    scale = (bounding_box_max - bounding_box_min).max() / 2.0
    return center, scale

if __name__ == "__main__":
    reconstruct_pyvista_primitive()
