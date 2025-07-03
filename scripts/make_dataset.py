import os
import h5py
import numpy as np
import pyvista as pv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

_BASE_PATH = Path("/mnt/storage01/workspace/research/gen11/shapenet_13")
_H5_Path = _BASE_PATH / "car_dataset"


def process_files(file_path: Path):
    print(f"Processing files in: {file_path}")
    files = file_path.rglob("*.obj")
    for pth in files:
        name = pth.parent.stem
        mesh = pv.read(pth)
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
        f.close()
    print(f"Finished processing files in: {file_path}")


if __name__ == "__main__":
    categories = list(_BASE_PATH.glob("*/"))

    os.makedirs(_H5_Path, exist_ok=True)
    os.makedirs(_H5_Path / "train", exist_ok=True)
    os.makedirs(_H5_Path / "test", exist_ok=True)

    with ProcessPoolExecutor() as exec:
        futures = exec.map(process_files, categories)
        for future in tqdm(as_completed(futures), total=len(categories)):
            pass

    print("All files processed and saved to HDF5 format.")
