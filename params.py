"""
Training parameters dataclass and serialization utilities.
"""

import dataclasses
from pathlib import Path

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
    IN_CHANNELS: int = 5     # default in paper is 25

    # Training parameters
    N_EPOCHS: int = 30       # default in paper is 100
    BATCH_SIZE: int = 16     # default in paper is 16    # TODO: wire this in
    LR: float = 5e-4         # default in paper is 5e-4
    PATIENCE: int = 10       # default in paper is 10    # TODO: wire this in

    # Cross-validation parameters
    N_FOLDS: int = 5         # default in paper is 5

    # Other parameters
    SEED: int = 42

    # logging/output parameters
    TRIAL_NAME: str = 'base-30-5folds'


def save_params_json(params: Params, output_path: Path) -> None:
    """Save Params dataclass as JSON to specified path."""
    save_dataclass_json(params, output_path)


def load_params_json(params_json: Path) -> Params:
    """Load Params dataclass from JSON file at specified path."""
    return load_dataclass_json(Params, params_json)
