"""
Training parameters dataclass and serialization utilities.
"""

import dataclasses
from pathlib import Path
from typing import Literal

from json_utils import save_dataclass_json, load_dataclass_json


@dataclasses.dataclass
class Params:
    """Training and model configuration parameters."""

    # Data and mode parameters
    VARIABLE: str = 'tmax'   # 'tmax' or 'precip'
    DATA_YEAR_START: int | None = 2023  # Year to start loading data from, None to include all data
                                        # Used for quicker testing

    # Model parameters
    N_CHANNELS: int = 128    # default in paper is 128
    N_BLOCKS: int = 6        # default in paper is 6
    KERNEL_SIZE: int = 5     # default in paper is 5
    LENGTH_SCALE: float = 0.1  # default in paper is 0.1
    IN_CHANNELS: int = 25     # default in paper is 25; this is best computed dynamically from the data available
    SEASONAL_FEATURES_IN_MLP: bool = True  # Whether to include seasonal features (cos/sin of day-of-year) in elevation MLP

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

    DEVICE: str = 'cpu'

    # Data paths (saved during training for reproducibility)
    ERA5_MAX_TEMP_GLOB: str | None = None
    ERA5_GEOPOTENTIAL_GLOB: str | None = None
    METEO_SWISS_MAX_TEMP_GLOB: str | None = None
    HI_RES_TOPOGRAPHY_ZARR_PATH: str | None = None


def save_params_json(params: Params, output_path: Path) -> None:
    """Save Params dataclass as JSON to specified path."""
    save_dataclass_json(params, output_path)


def load_params_json(params_json: Path) -> Params:
    """Load Params dataclass from JSON file at specified path."""
    return load_dataclass_json(Params, params_json)
