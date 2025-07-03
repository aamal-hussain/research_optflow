import numpy as np
import torch
from torch.utils.data import Dataset

from optflow.dora.dataset.sampling import (
    sample_coarse_points_and_normals,
    sample_points_and_signed_distance,
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
        num_input_points: int = 32_768,
        num_query_points: int = 16_384,
        minimum_sharp_edge_angle: float = 15.0,
    ):
        super().__init__()
        self._data = data
        self._mode = mode
        self._verts_key = verts_key
        self._faces_key = faces_key
        self._areas_key = areas_key
        self._faces_normals_key = faces_normals_key
        self._num_input_points = num_input_points
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

        coarse_pc, coarse_normals = sample_coarse_points_and_normals(
            sample[self._verts_key],
            sample[self._faces_key],
            sample[self._faces_normals_key],
            self._num_input_points,
        )
        sharp_pc, sharp_normals = sample_sharp_points_and_normals(
            sample[self._verts_key],
            sample[self._faces_key],
            sample[self._faces_normals_key],
            self._num_input_points,
            self._minimum_sharp_edge_angle,
        )
        encoder_sample.update(
            {
                "coarse_pc": coarse_pc,
                "coarse_feats": coarse_normals,
                "sharp_pc": sharp_pc,
                "sharp_feats": sharp_normals
            }
        )
        return encoder_sample

    def _decoder_sample(self, sample: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:

        coarse_xyz, coarse_sdf = sample_points_and_signed_distance(
            vertices=sample[self._verts_key],
            faces=sample[self._faces_key],
            num_samples=10_000,
            starting_points=sample["coarse_pc"],
            standard_deviations=[1e-3, 5e-3],
        )
        sharp_xyz, sharp_sdf = sample_points_and_signed_distance(
            vertices=sample[self._verts_key],
            faces=sample[self._faces_key],
            num_samples=21_384,
            starting_points=sample["sharp_pc"],
            standard_deviations=[1e-3, 5e-3, 7e-3, 1e-2],
        )
        agnostic_xyz, agnostic_sdf = sample_points_and_signed_distance(
            vertices=sample[self._verts_key],
            faces=sample[self._faces_key],
            num_samples=10_000
        )
        xyz=np.vstack((coarse_xyz, sharp_xyz, agnostic_xyz))
        sdf=np.concatenate((coarse_sdf, sharp_sdf, agnostic_sdf), axis=0)
        return {"query_points": np.asarray(xyz, dtype=np.float32), "sdf": np.asarray(sdf, dtype=np.float32)}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self._data[idx]
        sample = self._center_and_scale(sample)
        processed_sample = {}
        match self._mode:
            case InferenceMode.DEFAULT:
                processed_sample |= self._encoder_sample(processed_sample | sample)
                processed_sample |= self._decoder_sample(processed_sample | sample)
            case InferenceMode.ENCODER:
                processed_sample |= self._encoder_sample(processed_sample | sample)
            case InferenceMode.DECODER:
                processed_sample |= self._decoder_sample(processed_sample | sample)
            case _:
                raise ValueError(f"Unknown mode: {self._mode}")

        processed_sample = {
            k: torch.from_numpy(v).to(dtype=torch.float32) for k, v in processed_sample.items()
        }
        return processed_sample


def calculate_center_and_scale_of_mesh(verts: np.ndarray) -> tuple[np.ndarray, float]:
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    scale = (bbox_max - bbox_min).max() / 2.0
    return center, scale
