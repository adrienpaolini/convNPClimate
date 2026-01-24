"""Inference utilities for trained convCNP models."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import datasets as ds
import params
import model_factory
from convCNP.training.utils import get_sigma_tmax, get_value_tmax


@dataclass
class HoldoutFoldResult:
    """Results from predicting a holdout fold, denormalized to Celsius."""
    errors: np.ndarray   # (holdout_days, n_points)
    preds: np.ndarray    # (holdout_days, n_points)
    sigmas: np.ndarray   # (holdout_days, n_points)
    truths: np.ndarray   # (holdout_days, n_points)


def load_model_checkpoint(
    checkpoint_path: Path,
    p: params.Params,
    device: torch.device
) -> tuple[nn.Module, int]:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint file.
        p: Training parameters used to build the model architecture.
        device: Torch device to load the model onto.

    Returns:
        Tuple of (model, epoch) where epoch is the training epoch of the checkpoint.
    """
    # Build model architecture using shared factory
    model, _, _ = model_factory.build_model(p)
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

def predict_holdout_fold(
    model: nn.Module,
    context: torch.Tensor,
    holdout_start: int,
    holdout_end: int,
    dists: torch.Tensor,
    elev: torch.Tensor,
    seasonal: torch.Tensor | None,
    target_y: torch.Tensor,
    metadata: ds.Era5Metadata,
    device: torch.device,
) -> HoldoutFoldResult:
    """Predict all holdout days for one fold and return denormalized results.

    Args:
        model: Trained model in eval mode.
        context: ERA5 context data, shape (n_times, channels, lat, lon).
        holdout_start: Start index of the holdout period (inclusive).
        holdout_end: End index of the holdout period (exclusive).
        dists: Distance tensor, shape (n_points, lat, lon).
        elev: Elevation tensor, shape (n_points, n_elev_features).
        seasonal: Seasonal features tensor (n_times, 2) or None.
        target_y: Ground truth tensor, shape (n_times, n_points).
        metadata: Era5Metadata with denormalize method.
        device: Torch device.

    Returns:
        HoldoutFoldResult with errors, preds, sigmas, truths arrays,
        each shape (holdout_days, n_points), denormalized to Celsius.
    """
    fold_errors = []
    fold_preds = []
    fold_sigmas = []
    fold_truths = []

    for day_idx in range(holdout_start, holdout_end):
        day_preds, day_sigmas = predict_single_day(
            model, context, day_idx, dists, elev, seasonal, device
        )
        day_preds_np = day_preds.cpu().numpy()
        day_sigmas_np = day_sigmas.cpu().numpy()
        day_truth = target_y[day_idx].cpu().numpy()

        day_preds_denorm = metadata.denormalize(day_preds_np) - 273.15
        day_truth_denorm = metadata.denormalize(day_truth) - 273.15

        fold_errors.append(day_preds_denorm - day_truth_denorm)
        fold_preds.append(day_preds_denorm)
        fold_sigmas.append(day_sigmas_np)
        fold_truths.append(day_truth_denorm)

    return HoldoutFoldResult(
        errors=np.stack(fold_errors, axis=0),
        preds=np.stack(fold_preds, axis=0),
        sigmas=np.stack(fold_sigmas, axis=0),
        truths=np.stack(fold_truths, axis=0),
    )
