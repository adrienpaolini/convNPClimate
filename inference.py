"""Inference utilities for trained convCNP models."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

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

def compute_kriging_baseline(
    daily_tmax: 'pd.DataFrame',
    train_stations: list,
    test_stations: list,
    stations_meta: 'pd.DataFrame',
    dates_pd: 'pd.DatetimeIndex',
) -> np.ndarray:
    """
    Compute a regression kriging baseline for each day: predict test station
    tmax from train station observations using Universal Kriging with a
    regional linear drift (lat, lon as external drift terms).

    Returns:
        baseline: (n_days, n_test_stations) array of predictions in °C.
                  NaN where kriging cannot be computed (< 4 valid context obs).
    """
    try:
        from pykrige.uk import UniversalKriging
    except ImportError:
        raise ImportError("pykrige is required for the kriging baseline. "
                          "Run: pip install pykrige")

    train_meta = stations_meta.loc[train_stations]
    test_meta  = stations_meta.loc[test_stations]
    train_lats = train_meta['latitude'].values.astype(np.float64)
    train_lons = train_meta['longitude'].values.astype(np.float64)
    test_lats  = test_meta['latitude'].values.astype(np.float64)
    test_lons  = test_meta['longitude'].values.astype(np.float64)

    n_days  = len(dates_pd)
    n_test  = len(test_stations)
    baseline = np.full((n_days, n_test), np.nan, dtype=np.float32)

    for i, date in enumerate(dates_pd):
        obs = daily_tmax.loc[date, train_stations].values.astype(np.float64)
        valid = ~np.isnan(obs)
        if valid.sum() < 4:
            continue
        try:
            uk = UniversalKriging(
                train_lons[valid], train_lats[valid], obs[valid],
                variogram_model='spherical',
                drift_terms=['regional_linear'],
                verbose=False, enable_plotting=False,
            )
            preds, _ = uk.execute('points', test_lons, test_lats)
            baseline[i] = preds.astype(np.float32)
        except Exception:
            pass

    return baseline
