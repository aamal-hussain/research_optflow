import logging
import os
import h5py
import numpy as np
import pyvista as pv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

_BASE_PATH = Path("/mnt/storage01/workspace/research/gen11/shapenet_13")
_H5_Path = _BASE_PATH / "car_dataset"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_file(pth: Path):
    name = pth.parent.stem

    try:
        mesh = pv.read(pth)
    except Exception as e:
        logging.error(f"Failed to read mesh {pth}: {e}")
        raise

    if not mesh.is_manifold:
        return
    mesh = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False)
    mesh_data = {
        "mesh.verts": np.asarray(mesh.points),
        "mesh.faces": np.asarray(mesh.regular_faces),
        "mesh.verts_normals": np.asarray(mesh.point_normals),
        "mesh.faces_normals": np.asarray(mesh.cell_normals),
        "mesh.areas": np.asarray(mesh.cell_data["Area"]),
    }

    if np.random.rand() < 0.8:
        split = "train"
    else:
        split = "test"

    output_file_path = _H5_Path / split / f"{name}.h5"
    try:
        with h5py.File(output_file_path, "w") as f:
            for key, value in mesh_data.items():
                f.create_dataset(key, data=value)
    except Exception as e:
        logging.error(f"Failed to write HDF5 for {pth} to {output_file_path}: {e}")
        raise

    return


if __name__ == "__main__":
    os.makedirs(_H5_Path, exist_ok=True)
    os.makedirs(_H5_Path / "train", exist_ok=True)
    os.makedirs(_H5_Path / "test", exist_ok=True)

    _CAR_PATH = _BASE_PATH / "02958343"
    files = list(_CAR_PATH.rglob("*.obj"))
    logging.info(f"Found {len(files)} .obj files to process.")

    with ThreadPoolExecutor() as exec:
        futures = {exec.submit(process_file, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files)):
            file = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.info(f"FAILED processing {file}: {e}")
    logging.info("All files processed.")
