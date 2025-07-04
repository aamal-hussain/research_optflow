import os
import h5py
import numpy as np
import pyvista as pv
import pymeshfix as mf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

_BASE_PATH = Path("/mnt/storage01/workspace/research/gen11/shapenet_13")
_H5_Path = _BASE_PATH / "car_dataset"


def process_files(pth: Path):
    try:
        name = pth.parent.stem
        mesh = pv.read(pth)

        if not mesh.is_manifold:
            meshfix = mf.MeshFix(mesh)
            meshfix.repair(verbose=False)
            fixed_mesh = meshfix.mesh
            if fixed_mesh.is_manifold:
                mesh = fixed_mesh
            else:
                raise ValueError("Skipping this mesh as it is not manifold")


        mesh = mesh.compute_cell_sizes(length=False, area=True, volume=False)
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False)
        mesh_data = {
            "mesh.verts": mesh.points,
            "mesh.faces": mesh.regular_faces,
            "mesh.verts_normals": mesh.point_normals,
            "mesh.faces_normals": mesh.cell_normals,
            "mesh.areas": mesh.cell_data["Area"],
        }

        if np.random.rand() < 0.8:
            split = "train"
        else:
            split = "test"

        with h5py.File(_H5_Path / split / f"{name}.h5", "w") as f:
            for key, value in mesh_data.items():
                f.create_dataset(key, data=value)

    except Exception as e:
        raise ValueError(f"Error processing file {pth}: {e}") from e

    return

if __name__ == "__main__":

    os.makedirs(_H5_Path, exist_ok=True)
    os.makedirs(_H5_Path / "train", exist_ok=True)
    os.makedirs(_H5_Path / "test", exist_ok=True)

    _CAR_PATH = _BASE_PATH / "02958343"
    files = list(_CAR_PATH.rglob("*.obj"))

    with ThreadPoolExecutor() as exec:
        futures = {exec.submit(process_files, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures)):
            file = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"FAILED processing {file}: {e}")

    print("All files processed and saved to HDF5 format.")
