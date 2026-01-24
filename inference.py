"""Inference utilities for trained convCNP models."""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

import params
import model_factory
from convCNP.training.utils import get_sigma_tmax, get_value_tmax


def load_model_checkpoint(
    checkpoint_path: Path,
    params: params.Params,
    device: torch.device
) -> tuple[nn.Module, int]:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint file.
        params: Training parameters used to build the model architecture.
        device: Torch device to load the model onto.

    Returns:
        Tuple of (model, epoch) where epoch is the training epoch of the checkpoint.
    """
    # Build model architecture using shared factory
    model, _, _ = model_factory.build_model(params)
    model.to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle potential 'module.' prefix from DataParallel
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])

    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint.get('epoch', -1)


def predict_single_day(
    model: nn.Module,
    context: torch.Tensor,
    day_idx: int,
    dists: torch.Tensor,
    elev: torch.Tensor,
    seasonal: torch.Tensor | None,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate predictions for a single day.

    Args:
        model: The trained model.
        context: ERA5 context data, shape (n_times, channels, lat, lon).
        day_idx: Index of the day to predict.
        dists: Distance tensor, shape (n_points, lat, lon).
        elev: Elevation tensor, shape (n_points, n_elev_features).
        seasonal: Seasonal features tensor (n_times, 2) with [cos_doy, sin_doy], or None.
        device: Torch device.

    Returns:
        Tuple of (predictions, sigma) tensors, each of shape (n_points,).
    """
    model.eval()

    with torch.no_grad():
        # Get context for this day (add batch dimension)
        day_context = context[day_idx:day_idx+1]  # (1, channels, lat, lon)

        # Get seasonal features for this day if available
        day_seasonal = seasonal[day_idx:day_idx+1] if seasonal is not None else None  # (1, 2) or None

        # Create mask
        batch_size, channels, x, y = day_context.shape
        mask = torch.ones(batch_size, channels, x, y).to(device)

        # Forward pass with seasonal features (or None)
        output = model(day_context, mask, dists, elev, seasonal=day_seasonal)

        # Extract mean prediction (for tmax, output[:, :, 0] is the mean)
        predictions = get_value_tmax(output)  # (1, n_points)

        sigma = get_sigma_tmax(output)  # (1, n_points)

    return predictions.squeeze(0), sigma.squeeze(0)
