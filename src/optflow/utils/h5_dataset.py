from pathlib import Path

import h5py
import numpy as np


class H5Sample:
    def __init__(self, file: Path):
        self._file = file
        self._h5 = h5py.File(file, "r")

    def __call__(self) -> dict[str, np.ndarray]:
        sample = {}
        for key in self._h5.keys():
            sample[key] = self._h5[key][()]
        return sample


class H5Dataset:
    def __init__(self, base_path: Path):
        self._base_path = base_path
        self._files = list(base_path.glob("*.h5"))

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if idx < 0 or idx >= len(self._files):
            raise IndexError("Index out of bounds for H5Dataset")
        sample = H5Sample(self._files[idx])
        return sample()
