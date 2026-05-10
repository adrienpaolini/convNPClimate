"""Inference utilities for trained convCNP models."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import datasets as ds
from convCNP.training.utils import get_sigma_tmax, get_value_tmax


@dataclass
class HoldoutFoldResult:
    """Results from predicting a holdout fold, denormalized to Celsius."""
    errors: np.ndarray   # (holdout_days, n_points)
    preds: np.ndarray    # (holdout_days, n_points)
    sigmas: np.ndarray   # (holdout_days, n_points)
    truths: np.ndarray   # (holdout_days, n_points)


def predict_single_day_smacnp(
    model: nn.Module,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_target: torch.Tensor,
    day_idx: int,
    device: torch.device,
    chunk_size: int = 5000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate SMACNP predictions for a single day.

    Args:
        model: Trained SMACNP model.
        x_context: Context attributes, shape (T, N, input_dim).
        y_context: Context observations, shape (T, N, 1).
        x_target: Target attributes, shape (M, input_dim) or (T, M, input_dim).
        day_idx: Index of the day to predict.
        device: Torch device.
        chunk_size: Number of target points per forward pass (reduces peak memory).

    Returns:
        Tuple of (predictions, sigma), each shape (M,).
    """
    model.eval()

    xc = x_context[day_idx:day_idx + 1]  # (1, N, input_dim)
    yc = y_context[day_idx:day_idx + 1]  # (1, N, 1)
    xt_full = x_target.unsqueeze(0) if x_target.dim() == 2 else x_target[day_idx:day_idx + 1]
    # xt_full: (1, M, input_dim)

    M = xt_full.shape[1]
    pred_chunks, sigma_chunks = [], []

    with torch.no_grad():
        for start in range(0, M, chunk_size):
            xt_chunk = xt_full[:, start:start + chunk_size, :]  # (1, chunk, D)
            output = model(xc, yc, xt_chunk)                    # (1, chunk, 2)
            pred_chunks.append(get_value_tmax(output).squeeze(0))   # (chunk,)
            sigma_chunks.append(get_sigma_tmax(output).squeeze(0))  # (chunk,)

    return torch.cat(pred_chunks), torch.cat(sigma_chunks)


def predict_holdout_fold_smacnp(
    model: nn.Module,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_target: torch.Tensor,
    holdout_start: int,
    holdout_end: int,
    target_y: torch.Tensor,
    metadata: ds.Era5Metadata,
    device: torch.device,
    chunk_size: int = 5000,
) -> HoldoutFoldResult:
    """Predict all holdout days for one SMACNP fold.

    Args:
        model: Trained SMACNP model in eval mode.
        x_context: Context attributes, shape (T, N, input_dim).
        y_context: Context observations, shape (T, N, 1).
        x_target: Target attributes, shape (M, input_dim) or (T, M, input_dim).
        holdout_start: Start index of holdout period (inclusive).
        holdout_end: End index of holdout period (exclusive).
        target_y: Ground truth, shape (T, M).
        metadata: Era5Metadata with denormalize method.
        device: Torch device.
        chunk_size: Target points per forward pass.

    Returns:
        HoldoutFoldResult with errors, preds, sigmas, truths arrays,
        each shape (holdout_days, M), denormalized to Celsius.
    """
    fold_errors, fold_preds, fold_sigmas, fold_truths = [], [], [], []

    for day_idx in range(holdout_start, holdout_end):
        day_preds, day_sigmas = predict_single_day_smacnp(
            model, x_context, y_context, x_target, day_idx, device, chunk_size
        )
        day_preds_np = day_preds.cpu().numpy()
        day_sigmas_np = day_sigmas.cpu().numpy()
        day_truth = target_y[day_idx].cpu().numpy()

        day_preds_denorm = metadata.denormalize(day_preds_np) - ds.KELVIN_OFFSET
        day_truth_denorm = metadata.denormalize(day_truth) - ds.KELVIN_OFFSET

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
