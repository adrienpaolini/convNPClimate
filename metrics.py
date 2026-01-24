"""
Post-training evaluation metrics for the convNP climate downscaling model.
"""

import dataclasses
import warnings

import numpy as np
import properscoring as ps


@dataclasses.dataclass
class PerPixelMetrics:
    """Results of per-pixel metric computation over all holdout folds."""

    mae_grid: np.ndarray
    rmse_grid: np.ndarray
    bias_grid: np.ndarray
    crps_grid: np.ndarray
    mean_sigma_grid: np.ndarray
    skill_grid: np.ndarray
    overall_crps: float
    global_skill: float


def compute_perpixel_metrics(
    all_errors: np.ndarray,
    all_preds: np.ndarray,
    all_sigmas: np.ndarray,
    all_truths: np.ndarray,
    all_ref_preds: np.ndarray,
    grid_shape: tuple[int, int],
    valid_mask: np.ndarray,
) -> PerPixelMetrics:
    """Compute per-pixel evaluation metrics over combined holdout predictions.

    Args:
        all_errors: Prediction errors (pred - truth), shape (n_days, n_points), in Celsius.
        all_preds: Denormalized predictions, shape (n_days, n_points), in Celsius.
        all_sigmas: Predicted sigma values, shape (n_days, n_points).
        all_truths: Denormalized ground truth, shape (n_days, n_points), in Celsius.
        all_ref_preds: Reference (ERA5 interpolated) predictions, shape (n_days, n_points), in Celsius.
        grid_shape: (N, E) spatial grid dimensions.
        valid_mask: Boolean mask of valid pixels, shape (N, E).

    Returns:
        PerPixelMetrics dataclass with all grids masked and scalar summaries.
    """
    N, E = grid_shape

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

        # Per-pixel MAE, RMSE, bias, mean sigma
        mae_per_pixel = np.nanmean(np.abs(all_errors), axis=0)
        rmse_per_pixel = np.sqrt(np.nanmean(all_errors ** 2, axis=0))
        bias_per_pixel = np.nanmean(all_errors, axis=0)
        mean_sigma_per_pixel = np.nanmean(all_sigmas, axis=0)

        # CRPS (Gaussian predictions vs observations)
        crps_all = ps.crps_gaussian(all_truths, mu=all_preds, sig=all_sigmas)
        crps_per_pixel = np.nanmean(crps_all, axis=0)
        overall_crps = float(np.nanmean(crps_all))

        # Reference CRPS (deterministic ERA5 interpolation: CRPS = |pred - obs|)
        crps_ref_all = np.abs(all_ref_preds - all_truths)
        crps_ref_per_pixel = np.nanmean(crps_ref_all, axis=0)

    global_skill = skill_score(np.nanmean(crps_all), np.nanmean(crps_ref_all))
    skill_per_pixel = 1.0 - (crps_per_pixel / crps_ref_per_pixel)

    def _to_masked_grid(flat: np.ndarray) -> np.ndarray:
        return np.where(valid_mask, flat.reshape(N, E), np.nan)

    return PerPixelMetrics(
        mae_grid=_to_masked_grid(mae_per_pixel),
        rmse_grid=_to_masked_grid(rmse_per_pixel),
        bias_grid=_to_masked_grid(bias_per_pixel),
        crps_grid=_to_masked_grid(crps_per_pixel),
        mean_sigma_grid=_to_masked_grid(mean_sigma_per_pixel),
        skill_grid=_to_masked_grid(skill_per_pixel),
        overall_crps=overall_crps,
        global_skill=float(global_skill),
    )


def skill_score(crps_model, crps_reference):
    """
    Compute the skill score: 1 - (CRPS_model / CRPS_reference).

    Interpretation:
        1.0  = perfect model (zero CRPS)
        0.0  = model is as good as the reference
        <0   = model is worse than the reference

    Args:
        crps_model: mean CRPS of the probabilistic model predictions
        crps_reference: mean CRPS of the reference (for a deterministic
                        reference, CRPS equals the MAE)

    Returns:
        Skill score (scalar).
    """
    return 1.0 - (crps_model / crps_reference)
