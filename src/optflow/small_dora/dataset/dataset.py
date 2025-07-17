import numpy as np
import torch

from optflow.dora.dataset.dataset import DoraDataset
from optflow.dora.model import VAEMode
from optflow.utils.h5_dataset import H5Dataset


class SmallDoraDataset(DoraDataset):
    def __init__(
        self,
        data: H5Dataset,
        mode: VAEMode,
        verts_key: str = "mesh.verts",
        faces_key: str = "mesh.faces",
        areas_key: str = "mesh.areas",
        faces_normals_key: str = "mesh.face_normals",
        sdf_key: str = "sdf",
        sdf_gradients_key: str = "sdf_gradients",
        sdf_laplacian_key: str = "sdf_laplacian",
        query_key: str = "query_points",
        num_input_points: int = 32_768,
        num_query_points: int = 16_384,
        minimum_sharp_edge_angle: float = 15.0,
    ):
        super().__init__(data=data,
        mode=mode,
        verts_key=verts_key,
        faces_key=faces_key,
        areas_key=areas_key,
        faces_normals_key=faces_normals_key,
        num_input_points=num_input_points,
        num_query_points=num_query_points,
        minimum_sharp_edge_angle=minimum_sharp_edge_angle)

        self._sdf_key = sdf_key
        self._sdf_gradients_key = sdf_gradients_key
        self._sdf_laplacian_key = sdf_laplacian_key
        self._query_key = query_key


    def _decoder_sample(self, sample: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        if self._mode == VAEMode.DECODER:
            if {"latents", "query_points"}.issubset(sample.keys()):
                raise ValueError(
                    "In decoder mode, both the latents and query points must be passed"
                )
            elif not (
                isinstance(sample["latents"], np.ndarray)
                and isinstance(sample["query_points"], np.ndarray)
            ):
                raise TypeError(
                    f"latents and query_points must be of type np.ndarray, got {type(sample['latents'])} and {type(sample['query_points'])}"
                )
            else:
                return {
                    "latents": np.asarray(sample["latents"], dtype=np.float32),
                    "query_points": np.asarray(sample["query_points"], dtype=np.float32),
                }

        xyz = sample[self._query_key][:self._num_query_points]
        sdf = sample[self._sdf_key][:self._num_query_points]
        return {
            "query_points": np.asarray(xyz, dtype=np.float32),
            "sdf": np.asarray(sdf, dtype=np.float32),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self._data[idx]
        # sample = self._center_and_scale(sample)
        processed_sample = {}
        match self._mode:
            case VAEMode.DEFAULT:
                processed_sample |= self._encoder_sample(processed_sample | sample)
                processed_sample |= self._decoder_sample(processed_sample | sample)
            case VAEMode.ENCODER:
                processed_sample |= self._encoder_sample(processed_sample | sample)
            case VAEMode.DECODER:
                processed_sample |= self._decoder_sample(processed_sample | sample)
            case _:
                raise ValueError(f"Unknown mode: {self._mode}")

        processed_sample = {
            k: torch.from_numpy(v).to(dtype=torch.float32) for k, v in processed_sample.items()
        }
        return processed_sample | {"name": sample.get("name", "Unknown")}



