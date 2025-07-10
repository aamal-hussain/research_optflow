import torch
from torch.utils.data import Dataset

from optflow.utils.h5_dataset import H5Dataset


class LatentDataset(Dataset):
    def __init__(
        self,
        data: H5Dataset,
        mean_key: str = "latent.dora.2048.mean",
        logvar_key: str = "latent.dora.2048.log_variance",
        sample_posterior: bool = True,
    ):
        super().__init__()
        self._data = data
        self._mean_key = mean_key
        self._logvar_key = logvar_key
        self._sample_posterior = sample_posterior

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sample = self._data[idx]
        name = sample.get("name")
        mean = sample.get(self._mean_key)
        logvar = sample.get(self._logvar_key)

        if mean is None or logvar is None:
            raise ValueError(f"Sample {name} does not have {self._mean_key} or {self._logvar_key}")

        mean = torch.from_numpy(mean).to(dtype=torch.float32)
        if not self._sample_posterior:
            return mean
        logvar = torch.from_numpy(logvar).to(dtype=torch.float32)
        latent = mean + torch.exp(0.5 * logvar) * torch.rand_like(mean)
        return latent
