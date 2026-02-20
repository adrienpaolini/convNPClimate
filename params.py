"""
Training parameters dataclass and serialization utilities.
"""

import dataclasses
import os
import random
from pathlib import Path
from typing import Literal

import json_utils


@dataclasses.dataclass
class Params:
    """Training and model configuration parameters."""

    # Data and mode parameters
    VARIABLE: Literal['tmax', 'precip'] = 'tmax'
    DATA_YEAR_START: int | None = 2023  # Year to start loading data from, None to include all data
                                        # Used for quicker testing

    # Ablation parameters
    SEASONAL_FEATURES: bool = True  # Whether to include seasonal features (cos/sin of day-of-year) at all
    SEASONAL_FEATURES_IN_MLP: bool = True  # Whether to include seasonal features (cos/sin of day-of-year) in elevation MLP
    USE_ELEVATION: bool  = True  # Whether to include the elevation maps
    USE_MTPI: bool  = True  # Only used if USE_ELEVATION is also True

    # Model parameters
    N_CHANNELS: int = 128    # default in paper is 128
    N_BLOCKS: int = 6        # default in paper is 6
    KERNEL_SIZE: int = 5     # default in paper is 5
    LENGTH_SCALE: float = 0.1  # default in paper is 0.1
    IN_CHANNELS: int = 0     # default in paper is 25; this is best computed dynamically from the data available

    # Training parameters
    N_EPOCHS: int = 30       # default in paper is 100
    BATCH_SIZE: int = 16     # default in paper is 16
    LR: float = 5e-4         # default in paper is 5e-4
    PATIENCE: int = 10       # default in paper is 10

    # Cross-validation parameters
    N_FOLDS: int = 5         # default in paper is 5

    # Other parameters
    SEED: int = 42

    # logging/output parameters
    TRIAL_NAME: str = 'base-30-5folds'

    RUN_TYPE: Literal['local', 'cloud'] = 'local'

    DEVICE: str = 'cpu'  # Converted to torch.device in __post_init__

    # Data paths (saved during training for reproducibility)
    ERA5_MAX_TEMP_GLOB: str | None = None
    ERA5_GEOPOTENTIAL_GLOB: str | None = None
    METEO_SWISS_MAX_TEMP_GLOB: str | None = None
    HI_RES_TOPOGRAPHY_ZARR_PATH: str | None = None

    def __post_init__(self):
        assert self.N_EPOCHS > 0, "N_EPOCHS must be positive"
        assert self.BATCH_SIZE > 0, "BATCH_SIZE must be positive"
        assert self.N_FOLDS > 0, "N_FOLDS must be positive"
        assert self.LR > 0, "LR must be positive"
        assert self.VARIABLE in ('tmax', 'precip'), f"Unknown VARIABLE: {self.VARIABLE}"
        assert self.RUN_TYPE in ('local', 'cloud'), f"Unknown RUN_TYPE: {self.RUN_TYPE}"
        assert (not (not self.SEASONAL_FEATURES and self.SEASONAL_FEATURES_IN_MLP),
                "SEASONAL_FEATURES_IN_MLP cannot be True if SEASONAL_FEATURES (master switch) is False")


        import torch
        if isinstance(self.DEVICE, str):
            self.DEVICE = torch.device(self.DEVICE)

    def with_in_channels(self, in_channels: int) -> 'Params':
        """Return a new Params instance with IN_CHANNELS set."""
        return dataclasses.replace(self, IN_CHANNELS=in_channels)

    def with_data_paths(self, data_paths) -> 'Params':
        """Return a new Params instance with data paths populated."""
        return dataclasses.replace(
            self,
            ERA5_MAX_TEMP_GLOB=data_paths.ERA5_MAX_TEMP_GLOB,
            ERA5_GEOPOTENTIAL_GLOB=data_paths.ERA5_GEOPOTENTIAL_GLOB,
            METEO_SWISS_MAX_TEMP_GLOB=data_paths.METEO_SWISS_MAX_TEMP_GLOB,
            HI_RES_TOPOGRAPHY_ZARR_PATH=data_paths.HI_RES_TOPOGRAPHY_ZARR_PATH,
        )

    def save_json(self, output_path: Path) -> None:
        """Save this Params instance as JSON to specified path."""
        json_utils.save_dataclass_json(self, output_path)

    @staticmethod
    def load_json(params_json: Path) -> 'Params':
        """Load Params dataclass from JSON file at specified path."""
        return json_utils.load_dataclass_json(Params, params_json)


def is_renku() -> bool:
    return "RENKU_PROJECT_ID" in os.environ


def configure_renku_cuda() -> None:
    """Apply CUDA workarounds for Renku MIG (Multi-Instance GPU) partitions.

    The CUDA caching allocator's "expandable segments" feature requires NVML
    calls that fail on MIG devices. This disables that feature.
    """
    if "RENKU_PROJECT_ID" in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False,max_split_size_mb:512"


def select_device():
    """Select the best available torch device.

    Priority: CUDA > MPS > CPU.
    """
    # Note: we import torch only here, instead of doing it on the
    # top of the module to prevent this being imported before
    # configure_renku_cuda is executed, which would make CUDA
    # usage break.
    import torch

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across torch, numpy, Python, and CUDA."""
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
