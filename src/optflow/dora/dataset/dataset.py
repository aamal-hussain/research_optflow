import numpy as np
import torch
from torch.utils.data import Dataset

from optflow.dora.dataset.sampling import (
    sample_coarse_points_and_normals,
    sample_sharp_points_and_normals,
)
from optflow.dora.model import InferenceMode
from optflow.utils.h5_dataset import H5Dataset


class DoraDataset(Dataset):
    def __init__(
        self,
        data: H5Dataset,
        mode: InferenceMode,
        verts_key: str = "verts",
        faces_key: str = "faces",
        areas_key: str = "areas",
        faces_normals_key: str = "face_normals",
        num_coarse_points: int = 32_768,
        num_sharp_points: int = 32_768,
        num_query_points: int = 16_384,
        oversample_factor: int = 8,
        minimum_sharp_edge_angle: float = 15.0,
    ):
        super().__init__()
        self._data = data
        self._mode = mode
        self._verts_key = verts_key
        self._faces_key = faces_key
        self._areas_key = areas_key
        self._faces_normals_key = faces_normals_key
        self._num_coarse_points = num_coarse_points
        self._num_sharp_points = num_sharp_points
        self._oversample_factor = oversample_factor
        self._num_query_points = num_query_points
        self._minimum_sharp_edge_angle = minimum_sharp_edge_angle

    def __len__(self):
        return len(self._data)

    def _center_and_scale(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        center, scale = calculate_center_and_scale_of_mesh(sample[self._verts_key])
        sample[self._verts_key] = (sample[self._verts_key] - center) / scale
        sample[self._areas_key] = sample[self._areas_key] / (scale**2)
        return sample

    def _encoder_sample(self, sample: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        encoder_sample = {}
        points, normals = sample_coarse_points_and_normals(
            sample[self._verts_key],
            sample[self._faces_key],
            sample[self._faces_normals_key],
            num_samples=self._num_coarse_points,
            oversample_factor=self._oversample_factor,
        )
        encoder_sample.update(
            {
                "coarse_pc": torch.tensor(points, dtype=torch.float32),
                "coarse_feats": torch.tensor(normals, dtype=torch.float32),
            }
        )

        points, normals = sample_sharp_points_and_normals(
            sample[self._verts_key],
            sample[self._faces_key],
            sample[self._faces_normals_key],
            num_samples=self._num_coarse_points,
            oversample_factor=self._oversample_factor,
            minimum_sharp_edge_angle=self._minimum_sharp_edge_angle,
        )
        encoder_sample.update(
            {
                "sharp_pc": torch.tensor(points, dtype=torch.float32),
                "sharp_feats": torch.tensor(normals, dtype=torch.float32),
            }
        )
        return encoder_sample

    def _decoder_sample(self) -> dict[str, torch.Tensor]:
        return {"query_points": torch.randn((self._num_query_points, 3), dtype=torch.float32)}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self._data[idx]
        sample = self._center_and_scale(sample)
        processed_sample = {}
        match self._mode:
            case InferenceMode.DEFAULT:
                processed_sample |= self._encoder_sample(sample)
                processed_sample |= self._decoder_sample()
            case InferenceMode.ENCODER:
                processed_sample |= self._encoder_sample(sample)
            case InferenceMode.DECODER:
                processed_sample |= self._decoder_sample()
            case _:
                raise ValueError(f"Unknown mode: {self._mode}")
        return processed_sample


def calculate_center_and_scale_of_mesh(verts: np.ndarray) -> tuple[np.ndarray, float]:
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    scale = (bbox_max - bbox_min).max() / 2.0
    return center, scale
