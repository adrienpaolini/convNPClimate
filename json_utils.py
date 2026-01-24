"""
JSON serialization utilities for dataclasses with numpy array support.
"""

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar, Type
import numpy as np

T = TypeVar('T')


class _TorchDeviceEncoder(json.JSONEncoder):
    """JSON encoder that serializes torch.device objects as strings."""

    def default(self, o):
        import torch
        if isinstance(o, torch.device):
            return str(o)
        return super().default(o)


def save_dataclass_json(obj: T, output_path: Path, numpy_fields: list[str] | None = None) -> None:
    """
    Save a dataclass as JSON to the specified path.

    Args:
        obj: Dataclass instance to save
        output_path: Path to save JSON file
        numpy_fields: List of field names that contain numpy arrays (will be converted to lists)
    """
    obj_copy = copy.copy(obj)

    if numpy_fields:
        for field in numpy_fields:
            if hasattr(obj_copy, field):
                value = getattr(obj_copy, field)
                if isinstance(value, np.ndarray):
                    setattr(obj_copy, field, {"values": value.tolist(), "dtype": str(value.dtype)})

    with open(output_path, 'w') as f:
        json.dump(asdict(obj_copy), f, indent=4, cls=_TorchDeviceEncoder)


def load_dataclass_json(
    cls: Type[T],
    json_path: Path,
    numpy_fields: list[str] | None = None
) -> T:
    """
    Load a dataclass from a JSON file.

    Args:
        cls: The dataclass type to instantiate
        json_path: Path to the JSON file
        numpy_fields: List of field names that should be converted back to numpy arrays

    Returns:
        Instance of the dataclass
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if numpy_fields:
        for field in numpy_fields:
            if field in data and data[field] is not None:
                value = data[field]
                if isinstance(value, dict) and "values" in value and "dtype" in value:
                    data[field] = np.array(value["values"], dtype=np.dtype(value["dtype"]))
                else:
                    data[field] = np.array(value)

    return cls(**data)
