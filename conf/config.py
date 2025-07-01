from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from optflow.diffusion.noise_scheduler import ScheduleType

@dataclass
class DiffusionConfig:
    in_channels: int = 64
    inner_product_channels: int = 64
    num_heads: int = 8
    depth: int = 12
    num_freqs: int = 8
    include_pi: bool = True

@dataclass
class NoiseSchedulerConfig:
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_timesteps: int = 1000
    schedule_type: ScheduleType = ScheduleType.LINEAR


@dataclass
class Config:
    batch_size: int = 2
    sequence_length: int = 2048
    model: DiffusionConfig = field(default_factory=DiffusionConfig)
    noise_scheduler: NoiseSchedulerConfig = field(default_factory=NoiseSchedulerConfig)
    device: str = "cuda"

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
