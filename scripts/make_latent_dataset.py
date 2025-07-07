import logging
import os
import h5py
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

_BASE_PATH = Path("/mnt/storage01/workspace/research/opora/experiments/luminary_300k_v5/02_processed_data")
_LATENT_PATH = Path("data/luminary_latents")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_file(pth: Path):
    name = pth.stem
    try:
        sample = {}
        with h5py.File(pth, "r") as f:
            for key in {"latent.dora.2048.mean", "latent.dora.2048.log_variance"}:
                sample[key] = f[key][()]
    except Exception as e:
        logging.error(f"Failed to read mesh {pth}: {e}")
        raise

    if np.random.rand() < 0.8:
        split = "train"
    else:
        split = "test"

    output_file_path = _LATENT_PATH / split / f"{name}.h5"
    try:
        with h5py.File(output_file_path, "w") as f:
            for key, value in sample.items():
                f.create_dataset(key, data=value)
    except Exception as e:
        logging.error(f"Failed to write HDF5 for {pth} to {output_file_path}: {e}")
        raise

    return

if __name__ == "__main__":

    os.makedirs(_LATENT_PATH, exist_ok=True)
    os.makedirs(_LATENT_PATH / "train", exist_ok=True)
    os.makedirs(_LATENT_PATH / "test", exist_ok=True)

    files = list(_BASE_PATH.glob("*.h5"))
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
