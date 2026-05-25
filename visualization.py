"""Visualization utilities for convCNP prediction analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from scipy.ndimage import zoom

import datasets as ds


def _save_fig(fig, save_path: str | Path | None):
    """Save figure to disk if save_path is provided."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {save_path}")


def plot_prediction_comparison(
    truth: np.ndarray,
    predictions: np.ndarray,
    era5_input: np.ndarray,
    grid_shape: tuple[int, int],
    era5_shape: tuple[int, int],
    metadata: ds.Era5Metadata,
    title: str,
    denormalize: bool = True,
    temp_range: tuple[float, float] = (-10, 30),
    residual_range: tuple[float, float] = (-10, 10),
    save_path: str | Path | None = None,
):
    """
    Plot side-by-side heatmaps: ERA5 Input | Ground Truth | Predictions | Residuals

    Args:
        truth: Ground truth values (n_points,)
        predictions: Model predictions (n_points,)
        era5_input: ERA5 temperature input (lat, lon) - normalized
        grid_shape: (N, E) shape for reshaping MeteoSwiss data
        era5_shape: (lat, lon) shape of ERA5 grid
        metadata: Era5Metadata for denormalization
        title: Plot title
        denormalize: If True, convert from normalized to Kelvin then Celsius
        temp_range: (min, max) temperature range in °C for truth/predictions
        residual_range: (min, max) range in °C for residuals
    """
    # Reshape to 2D grid
    N, E = grid_shape
    truth_grid = truth.reshape(N, E)
    pred_grid = predictions.reshape(N, E)

    # Mask predictions to only show where ground truth is valid
    valid_mask = ~np.isnan(truth_grid)
    pred_grid = np.where(valid_mask, pred_grid, np.nan)

    # Upsample ERA5 to MeteoSwiss resolution and apply same mask
    era5_lat, era5_lon = era5_shape
    zoom_factors = (N / era5_lat, E / era5_lon)
    era5_upsampled = zoom(era5_input, zoom_factors, order=0)  # nearest neighbour interpolation
    # ERA5 NetCDF files store latitude in decreasing order (north to south), so the first row
    # era5_data[day_idx, 0][0, :] is the northern edge. When we use origin='lower' with imshow,
    # row 0 is placed at the bottom, putting north at the bottom instead of the top. We could
    # go the other way, but then we would need to flip the mask too.
    era5_upsampled = np.flipud(era5_upsampled)
    era5_upsampled = np.where(valid_mask, era5_upsampled, np.nan)

    # Denormalize if requested
    if denormalize:
        # Convert from normalized to Kelvin
        truth_grid = metadata.denormalize(truth_grid)
        pred_grid = metadata.denormalize(pred_grid)
        era5_upsampled = metadata.denormalize(era5_upsampled)
        # Convert Kelvin to Celsius
        truth_grid = truth_grid - ds.KELVIN_OFFSET
        pred_grid = pred_grid - ds.KELVIN_OFFSET
        era5_upsampled = era5_upsampled - ds.KELVIN_OFFSET
        unit = '°C'
    else:
        unit = 'normalized'

    # Calculate residuals
    residuals_grid = pred_grid - truth_grid

    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Use fixed colormap ranges
    vmin, vmax = temp_range
    res_min, res_max = residual_range

    # ERA5 Input
    im0 = axes[0].imshow(era5_upsampled, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title(f'Input ({unit})')
    axes[0].set_xlabel('E (grid index)')
    axes[0].set_ylabel('N (grid index)')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Ground Truth
    im1 = axes[1].imshow(truth_grid, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title(f'Ground Truth ({unit})')
    axes[1].set_xlabel('E (grid index)')
    axes[1].set_ylabel('N (grid index)')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Predictions
    im2 = axes[2].imshow(pred_grid, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, origin='lower')
    axes[2].set_title(f'Predictions ({unit})')
    axes[2].set_xlabel('E (grid index)')
    axes[2].set_ylabel('N (grid index)')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    # Residuals (Bias)
    im3 = axes[3].imshow(residuals_grid, cmap='RdBu_r', vmin=res_min, vmax=res_max, origin='lower')
    axes[3].set_title(f'Bias (Pred - Truth, {unit})')
    axes[3].set_xlabel('E (grid index)')
    axes[3].set_ylabel('N (grid index)')
    plt.colorbar(im3, ax=axes[3], shrink=0.8)

    # Add statistics to residuals plot
    valid_residuals = residuals_grid[~np.isnan(residuals_grid)]
    mae = np.mean(np.abs(valid_residuals))
    rmse = np.sqrt(np.mean(valid_residuals**2))
    bias = np.mean(valid_residuals)
    stats_text = f'MAE: {mae:.2f}{unit}\nRMSE: {rmse:.2f}{unit}\nBias: {bias:.2f}{unit}'
    axes[3].text(0.02, 0.98, stats_text, transform=axes[3].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, save_path)

    return fig


def plot_error_maps(
    mae_grid: np.ndarray,
    rmse_grid: np.ndarray,
    bias_grid: np.ndarray,
    title: str,
    error_range: tuple[float, float] = (0, 5),
    bias_range: tuple[float, float] = (-3, 3),
    save_path: str | Path | None = None,
):
    """
    Plot per-pixel error magnitude maps: MAE | RMSE | Mean Bias

    Args:
        mae_grid: Per-pixel MAE values (N, E)
        rmse_grid: Per-pixel RMSE values (N, E)
        bias_grid: Per-pixel mean bias values (N, E)
        title: Plot title
        error_range: (min, max) range for MAE/RMSE colorbar
        bias_range: (min, max) range for bias colorbar
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    err_min, err_max = error_range
    bias_min, bias_max = bias_range

    # MAE Map
    im0 = axes[0].imshow(mae_grid, cmap='YlOrRd', vmin=err_min, vmax=err_max, origin='lower')
    axes[0].set_title('Per-Pixel MAE (°C)')
    axes[0].set_xlabel('E (grid index)')
    axes[0].set_ylabel('N (grid index)')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Global MAE
    valid_mae = mae_grid[~np.isnan(mae_grid)]
    mean_mae = np.mean(valid_mae)
    axes[0].text(0.02, 0.98, f'Mean: {mean_mae:.2f}°C', transform=axes[0].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # RMSE Map
    im1 = axes[1].imshow(rmse_grid, cmap='YlOrRd', vmin=err_min, vmax=err_max, origin='lower')
    axes[1].set_title('Per-Pixel RMSE (°C)')
    axes[1].set_xlabel('E (grid index)')
    axes[1].set_ylabel('N (grid index)')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Overall RMSE
    valid_rmse = rmse_grid[~np.isnan(rmse_grid)]
    mean_rmse = np.mean(valid_rmse)
    axes[1].text(0.02, 0.98, f'Mean: {mean_rmse:.2f}°C', transform=axes[1].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Mean Bias Map
    im2 = axes[2].imshow(bias_grid, cmap='RdBu_r', vmin=bias_min, vmax=bias_max, origin='lower')
    axes[2].set_title('Per-Pixel Mean Bias (°C)')
    axes[2].set_xlabel('E (grid index)')
    axes[2].set_ylabel('N (grid index)')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    # Overall bias
    valid_bias = bias_grid[~np.isnan(bias_grid)]
    mean_bias = np.mean(valid_bias)
    axes[2].text(0.02, 0.98, f'Mean: {mean_bias:.2f}°C', transform=axes[2].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, save_path)

    return fig


def plot_uncertainty_vs_error(
    sigma_grid: np.ndarray,
    abs_error_grid: np.ndarray,
    title: str,
    sigma_range: tuple[float, float] = (0, 5),
    error_range: tuple[float, float] = (0, 10),
    save_path: str | Path | None = None,
):
    """
    Plot uncertainty (sigma) vs absolute error to check if uncertainty correlates with errors.

    Ideally, high sigma values should appear in the same locations as high errors,
    indicating the model "knows" where it is uncertain.

    Args:
        sigma_grid: Predicted sigma values (N, E) in °C
        abs_error_grid: Absolute error |pred - truth| (N, E) in °C
        title: Plot title
        sigma_range: (min, max) range for sigma colorbar
        error_range: (min, max) range for absolute error colorbar
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sig_min, sig_max = sigma_range
    err_min, err_max = error_range

    # Uncertainty (Sigma) Map
    im0 = axes[0].imshow(sigma_grid, cmap='Purples', vmin=sig_min, vmax=sig_max, origin='lower')
    axes[0].set_title('Predicted Uncertainty (σ, °C)')
    axes[0].set_xlabel('E (grid index)')
    axes[0].set_ylabel('N (grid index)')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    valid_sigma = sigma_grid[~np.isnan(sigma_grid)]
    mean_sigma = np.mean(valid_sigma)
    axes[0].text(0.02, 0.98, f'Mean σ: {mean_sigma:.2f}°C', transform=axes[0].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Absolute Error Map
    im1 = axes[1].imshow(abs_error_grid, cmap='YlOrRd', vmin=err_min, vmax=err_max, origin='lower')
    axes[1].set_title('Absolute Error (|Pred - Truth|, °C)')
    axes[1].set_xlabel('E (grid index)')
    axes[1].set_ylabel('N (grid index)')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    valid_error = abs_error_grid[~np.isnan(abs_error_grid)]
    mean_error = np.mean(valid_error)
    axes[1].text(0.02, 0.98, f'Mean: {mean_error:.2f}°C', transform=axes[1].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Scatter plot: Sigma vs Absolute Error
    valid_mask = ~np.isnan(sigma_grid) & ~np.isnan(abs_error_grid)
    sigma_flat = sigma_grid[valid_mask]
    error_flat = abs_error_grid[valid_mask]

    # Subsample for scatter plot if too many points
    n_points = len(sigma_flat)
    if n_points > 5000:
        idx = np.random.choice(n_points, 5000, replace=False)
        sigma_sample = sigma_flat[idx]
        error_sample = error_flat[idx]
    else:
        sigma_sample = sigma_flat
        error_sample = error_flat

    axes[2].scatter(sigma_sample, error_sample, alpha=0.3, s=5, c='steelblue')
    axes[2].set_xlabel('Predicted σ (°C)')
    axes[2].set_ylabel('Absolute Error (°C)')
    axes[2].set_title('Uncertainty vs Error Correlation')

    # Add correlation coefficient
    correlation = np.corrcoef(sigma_flat, error_flat)[0, 1]
    axes[2].text(0.02, 0.98, f'Correlation: {correlation:.3f}', transform=axes[2].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, save_path)

    return fig


def plot_single_day_uncertainty(
    predictions: np.ndarray,
    sigma: np.ndarray,
    truth: np.ndarray,
    grid_shape: tuple[int, int],
    metadata: ds.Era5Metadata,
    title: str,
    save_path: str | Path | None = None,
):
    """Plot uncertainty vs error for a single day's predictions.

    Denormalizes predictions and truth to Celsius, computes absolute error,
    reshapes to the spatial grid, and delegates to plot_uncertainty_vs_error.

    Args:
        predictions: Normalized predictions, shape (n_points,).
        sigma: Predicted sigma values, shape (n_points,).
        truth: Normalized ground truth, shape (n_points,).
        grid_shape: (N, E) spatial grid dimensions.
        metadata: Era5Metadata with denormalize method.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    N, E = grid_shape

    pred_denorm = metadata.denormalize(predictions) - ds.KELVIN_OFFSET
    truth_denorm = metadata.denormalize(truth) - ds.KELVIN_OFFSET

    abs_error = np.abs(pred_denorm - truth_denorm)

    sigma_grid = sigma.reshape(N, E)
    abs_error_grid = abs_error.reshape(N, E)

    # Mask pixels outside the valid MeteoSwiss domain
    valid_mask = ~np.isnan(truth.reshape(N, E))
    sigma_grid = np.where(valid_mask, sigma_grid, np.nan)
    abs_error_grid = np.where(valid_mask, abs_error_grid, np.nan)

    max_sigma = np.nanmax(sigma_grid)
    max_error = np.nanmax(abs_error_grid)

    return plot_uncertainty_vs_error(
        sigma_grid=sigma_grid,
        abs_error_grid=abs_error_grid,
        title=title,
        sigma_range=(0, max_sigma),
        error_range=(0, max_error),
        save_path=save_path,
    )


def plot_crps_map(
    crps_grid: np.ndarray,
    title: str,
    crps_range: tuple[float, float] = (0, 3),
    save_path: str | Path | None = None,
):
    """
    Plot per-pixel CRPS map.

    Args:
        crps_grid: Per-pixel mean CRPS values (N, E)
        title: Plot title
        crps_range: (min, max) range for CRPS colorbar
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    crps_min, crps_max = crps_range

    im = ax.imshow(crps_grid, cmap='YlOrRd', vmin=crps_min, vmax=crps_max, origin='lower')
    ax.set_title('Per-Pixel Mean CRPS (°C)')
    ax.set_xlabel('E (grid index)')
    ax.set_ylabel('N (grid index)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Add overall CRPS statistic
    valid_crps = crps_grid[~np.isnan(crps_grid)]
    mean_crps = np.mean(valid_crps)
    ax.text(0.02, 0.98, f'Overall CRPS: {mean_crps:.3f}°C', transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, save_path)

    return fig, mean_crps


def plot_skill_map(
    skill_grid: np.ndarray,
    global_skill: float,
    title: str,
    skill_range: tuple[float, float] = (-1, 1),
    save_path: str | Path | None = None,
):
    """
    Plot per-pixel skill score map.

    Skill score = 1 - (CRPS_model / CRPS_reference).
    Positive = model beats ERA5 interpolation; negative = model is worse.

    Args:
        skill_grid: Per-pixel skill scores (N, E)
        global_skill: Global (spatially-averaged) skill score
        title: Plot title
        skill_range: (min, max) range for colorbar
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    skill_min, skill_max = skill_range

    im = ax.imshow(skill_grid, cmap='RdYlGn', vmin=skill_min, vmax=skill_max, origin='lower')
    ax.set_title('Per-Pixel Skill Score')
    ax.set_xlabel('E (grid index)')
    ax.set_ylabel('N (grid index)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Add statistics
    valid_skill = skill_grid[~np.isnan(skill_grid)]
    median_skill = np.median(valid_skill)
    stats_text = f'Global skill: {global_skill:.3f}\nMedian pixel skill: {median_skill:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, save_path)

    return fig


def plot_perpixel_uncertainty_vs_error(
    mean_sigma_grid: np.ndarray,
    mae_grid: np.ndarray,
    title: str,
    sigma_range: tuple[float, float] = (0, 5),
    error_range: tuple[float, float] = (0, 5),
    save_path: str | Path | None = None,
):
    """
    Plot per-pixel mean uncertainty (sigma) vs per-pixel MAE across all holdout days.

    Args:
        mean_sigma_grid: Per-pixel mean predicted sigma (N, E) in °C
        mae_grid: Per-pixel MAE (N, E) in °C
        title: Plot title
        sigma_range: (min, max) range for sigma colorbar
        error_range: (min, max) range for MAE colorbar
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sig_min, sig_max = sigma_range
    err_min, err_max = error_range

    # Mean Uncertainty (Sigma) Map
    im0 = axes[0].imshow(mean_sigma_grid, cmap='Purples', vmin=sig_min, vmax=sig_max, origin='lower')
    axes[0].set_title('Per-Pixel Mean Uncertainty (σ, °C)')
    axes[0].set_xlabel('E (grid index)')
    axes[0].set_ylabel('N (grid index)')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    valid_sigma = mean_sigma_grid[~np.isnan(mean_sigma_grid)]
    overall_mean_sigma = np.mean(valid_sigma)
    axes[0].text(0.02, 0.98, f'Mean σ: {overall_mean_sigma:.2f}°C', transform=axes[0].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # MAE Map
    im1 = axes[1].imshow(mae_grid, cmap='YlOrRd', vmin=err_min, vmax=err_max, origin='lower')
    axes[1].set_title('Per-Pixel MAE (°C)')
    axes[1].set_xlabel('E (grid index)')
    axes[1].set_ylabel('N (grid index)')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    valid_mae = mae_grid[~np.isnan(mae_grid)]
    overall_mae = np.mean(valid_mae)
    axes[1].text(0.02, 0.98, f'Mean: {overall_mae:.2f}°C', transform=axes[1].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Scatter plot: Mean Sigma vs MAE (per pixel)
    valid_mask = ~np.isnan(mean_sigma_grid) & ~np.isnan(mae_grid)
    sigma_flat = mean_sigma_grid[valid_mask]
    mae_flat = mae_grid[valid_mask]

    # Subsample for scatter plot if too many points
    n_points = len(sigma_flat)
    if n_points > 5000:
        idx = np.random.choice(n_points, 5000, replace=False)
        sigma_sample = sigma_flat[idx]
        mae_sample = mae_flat[idx]
    else:
        sigma_sample = sigma_flat
        mae_sample = mae_flat

    axes[2].scatter(sigma_sample, mae_sample, alpha=0.3, s=5, c='steelblue')
    axes[2].set_xlabel('Per-Pixel Mean σ (°C)')
    axes[2].set_ylabel('Per-Pixel MAE (°C)')
    axes[2].set_title('Uncertainty vs Error Correlation (per pixel)')

    # Add regression line
    slope, intercept = np.polyfit(sigma_flat, mae_flat, 1)
    x_line = np.linspace(np.min(sigma_flat), np.max(sigma_flat), 100)
    axes[2].plot(x_line, slope * x_line + intercept, 'r:', linewidth=2)

    # Add correlation coefficient
    correlation = np.corrcoef(sigma_flat, mae_flat)[0, 1]
    axes[2].text(0.02, 0.98, f'Correlation: {correlation:.3f}', transform=axes[2].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, save_path)

    return fig, correlation


def plot_perpixel_correlation(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    x_cmap: str = 'viridis',
    y_cmap: str = 'YlOrRd',
    save_path: str | Path | None = None,
):
    """
    Generic function to plot per-pixel correlation between two variables.

    Args:
        x_grid: First variable grid (N, E)
        y_grid: Second variable grid (N, E)
        x_label: Label for x variable
        y_label: Label for y variable
        title: Plot title
        x_range: (min, max) range for x colorbar
        y_range: (min, max) range for y colorbar
        x_cmap: Colormap for x variable
        y_cmap: Colormap for y variable
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Auto-compute ranges if not provided
    if x_range is None:
        x_range = (np.nanmin(x_grid), np.nanmax(x_grid))
    if y_range is None:
        y_range = (np.nanmin(y_grid), np.nanmax(y_grid))

    x_min, x_max = x_range
    y_min, y_max = y_range

    # X variable Map
    im0 = axes[0].imshow(x_grid, cmap=x_cmap, vmin=x_min, vmax=x_max, origin='lower')
    axes[0].set_title(x_label)
    axes[0].set_xlabel('E (grid index)')
    axes[0].set_ylabel('N (grid index)')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    valid_x = x_grid[~np.isnan(x_grid)]
    mean_x = np.mean(valid_x)
    axes[0].text(0.02, 0.98, f'Mean: {mean_x:.2f}', transform=axes[0].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Y variable Map
    im1 = axes[1].imshow(y_grid, cmap=y_cmap, vmin=y_min, vmax=y_max, origin='lower')
    axes[1].set_title(y_label)
    axes[1].set_xlabel('E (grid index)')
    axes[1].set_ylabel('N (grid index)')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    valid_y = y_grid[~np.isnan(y_grid)]
    mean_y = np.mean(valid_y)
    axes[1].text(0.02, 0.98, f'Mean: {mean_y:.2f}', transform=axes[1].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Scatter plot
    valid_mask = ~np.isnan(x_grid) & ~np.isnan(y_grid)
    x_flat = x_grid[valid_mask]
    y_flat = y_grid[valid_mask]

    # Subsample for scatter plot if too many points
    n_points = len(x_flat)
    if n_points > 5000:
        idx = np.random.choice(n_points, 5000, replace=False)
        x_sample = x_flat[idx]
        y_sample = y_flat[idx]
    else:
        x_sample = x_flat
        y_sample = y_flat

    axes[2].scatter(x_sample, y_sample, alpha=0.3, s=5, c='steelblue')
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel(y_label)
    axes[2].set_title(f'{x_label} vs {y_label}')

    # Add regression line
    slope, intercept = np.polyfit(x_flat, y_flat, 1)
    x_line = np.linspace(np.min(x_flat), np.max(x_flat), 100)
    axes[2].plot(x_line, slope * x_line + intercept, 'r:', linewidth=2)

    # Add correlation coefficient
    correlation = np.corrcoef(x_flat, y_flat)[0, 1]
    axes[2].text(0.02, 0.98, f'Correlation: {correlation:.3f}', transform=axes[2].transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, save_path)

    return fig, correlation


def plot_temporal_error_analysis(
    all_errors: np.ndarray,
    holdout_dates: np.ndarray,
    title: str,
    save_path: str | Path | None = None,
):
    """
    Plot day of year vs mean absolute error scatter with correlation.

    Args:
        all_errors: Array of shape (holdout_days, n_points) with errors for each day/pixel
        holdout_dates: Array of datetime64 values for each holdout day
        title: Plot title

    Returns:
        fig: matplotlib figure
        correlation: Correlation between day of year and mean absolute error
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Calculate mean absolute error per day (across all pixels)
    mae_per_day = np.nanmean(np.abs(all_errors), axis=1)

    # Convert datetime64 to day of year (1-365/366)
    dates_series = pd.to_datetime(holdout_dates)
    days_of_year = dates_series.dayofyear.values

    # Scatter plot with correlation
    ax.scatter(days_of_year, mae_per_day, alpha=0.6, s=30, c='steelblue')
    ax.set_xlabel('Month')
    ax.set_ylabel('Mean Absolute Error (°C)')
    ax.set_title('Month vs MAE Correlation')
    ax.set_xlim(1, 366)

    # Major ticks at month starts (tick marks only, no labels)
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ax.xaxis.set_major_locator(FixedLocator(month_starts))
    ax.xaxis.set_major_formatter(NullFormatter())

    # Minor ticks at month midpoints (labels only, no tick marks)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_midpoints = [(month_starts[i] + month_starts[i + 1]) / 2
                       for i in range(11)] + [(335 + 366) / 2]
    ax.xaxis.set_minor_locator(FixedLocator(month_midpoints))
    ax.xaxis.set_minor_formatter(FixedFormatter(month_names))
    ax.tick_params(axis='x', which='minor', length=0)

    # Add regression line
    slope, intercept = np.polyfit(days_of_year, mae_per_day, 1)
    x_line = np.linspace(1, 366, 100)
    ax.plot(x_line, slope * x_line + intercept, 'r:', linewidth=2)

    # Add correlation coefficient
    correlation = np.corrcoef(days_of_year, mae_per_day)[0, 1]
    ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, save_path)

    return fig, correlation


def plot_qq_calibration(
    all_truths: np.ndarray,
    all_preds: np.ndarray,
    all_sigmas: np.ndarray,
    title: str = "Q-Q Plot: Predicted vs Observed Distribution",
    n_quantiles: int = 100,
    save_path: str | Path | None = None,
):
    """
    Plot Q-Q plot comparing predicted distribution quantiles to observed quantiles.

    For each prediction, the model outputs N(μ, σ). We compute quantiles of this
    predicted distribution and compare them to the actual observed quantiles of
    the truth values. If calibrated, points should lie on the diagonal.

    This uses the Probability Integral Transform (PIT): for well-calibrated
    predictions, CDF(truth | μ, σ) should be uniform on [0, 1].

    Args:
        all_truths: Ground truth values, shape (n_days, n_points) or flattened
        all_preds: Predicted means (same shape as all_truths)
        all_sigmas: Predicted standard deviations (same shape as all_truths)
        title: Plot title
        n_quantiles: Number of quantile points to plot

    Returns:
        fig: matplotlib Figure
        calibration_stats: dict with calibration statistics
    """
    from scipy import stats as scipy_stats

    # Flatten arrays
    truths_flat = all_truths.flatten()
    preds_flat = all_preds.flatten()
    sigmas_flat = all_sigmas.flatten()

    # Remove NaN and invalid sigma values
    valid_mask = (
        ~np.isnan(truths_flat) &
        ~np.isnan(preds_flat) &
        ~np.isnan(sigmas_flat) &
        (sigmas_flat > 0)
    )
    truths_valid = truths_flat[valid_mask]
    preds_valid = preds_flat[valid_mask]
    sigmas_valid = sigmas_flat[valid_mask]

    n_valid = len(truths_valid)

    # Compute PIT values: CDF of truth under predicted distribution N(pred, sigma)
    # PIT = Φ((truth - pred) / sigma) where Φ is the standard normal CDF
    pit_values = scipy_stats.norm.cdf(truths_valid, loc=preds_valid, scale=sigmas_valid)

    # Define probability levels for Q-Q plot
    probs = np.linspace(0.01, 0.99, n_quantiles)

    # Theoretical quantiles: if well-calibrated, PIT ~ Uniform(0,1)
    # So theoretical quantiles are just the probability levels themselves
    theoretical_quantiles = probs

    # Empirical quantiles of PIT values
    empirical_quantiles = np.percentile(pit_values, probs * 100)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot Q-Q points
    ax.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.7, s=30, c='steelblue',
               label='PIT quantiles')

    # Plot reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'r-', linewidth=2, label='Perfect calibration')

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Theoretical Quantiles (Uniform)')
    ax.set_ylabel('Empirical Quantiles (PIT values)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Compute calibration statistics
    # 1. KS test: are PIT values uniform?
    ks_stat, ks_p = scipy_stats.kstest(pit_values, 'uniform')

    # 2. Coverage statistics using the PIT values
    # If well-calibrated, X% of PIT values should be <= X for any X
    coverage_50 = np.mean((pit_values >= 0.25) & (pit_values <= 0.75))  # Central 50%
    coverage_90 = np.mean((pit_values >= 0.05) & (pit_values <= 0.95))  # Central 90%

    # 3. Mean and std of PIT (should be 0.5 and ~0.289 for uniform)
    pit_mean = np.mean(pit_values)
    pit_std = np.std(pit_values)

    # 4. Also compute z-score statistics for reference
    z = (truths_valid - preds_valid) / sigmas_valid
    z_std = np.std(z)

    calibration_stats = {
        'pit_mean': pit_mean,        # Should be ~0.5
        'pit_std': pit_std,          # Should be ~0.289 (1/sqrt(12))
        'coverage_50': coverage_50,  # Should be ~0.50
        'coverage_90': coverage_90,  # Should be ~0.90
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'z_std': z_std,              # Should be ~1 if calibrated
        'n_valid': n_valid
    }

    # Add statistics text box
    stats_text = (
        f'n = {n_valid:,}\n'
        f'PIT mean: {pit_mean:.3f} (ideal: 0.5)\n'
        f'PIT std: {pit_std:.3f} (ideal: 0.289)\n'
        f'Central 50% coverage: {coverage_50:.1%} (ideal: 50%)\n'
        f'Central 90% coverage: {coverage_90:.1%} (ideal: 90%)\n'
        f'z std: {z_std:.2f} (ideal: 1.0)\n'
        f'KS p-value: {ks_p:.3g}'
    )
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    _save_fig(fig, save_path)

    return fig, calibration_stats


def plot_reliability_diagram(
    all_truths: np.ndarray,
    all_preds: np.ndarray,
    all_sigmas: np.ndarray,
    title: str = "Reliability Diagram: Predicted vs Observed Coverage",
    n_bins: int = 20,
    save_path: str | Path | None = None,
):
    """
    Plot reliability diagram showing predicted confidence vs observed frequency.

    For a well-calibrated probabilistic model, when we predict an X% confidence
    interval, the true value should fall within that interval X% of the time.

    This plot shows:
    - X-axis: Predicted confidence level (e.g., 50%, 80%, 90%, 95%)
    - Y-axis: Observed frequency (fraction of truths within that interval)

    Args:
        all_truths: Ground truth values, shape (n_days, n_points) or flattened
        all_preds: Predicted means (same shape as all_truths)
        all_sigmas: Predicted standard deviations (same shape as all_truths)
        title: Plot title
        n_bins: Number of confidence levels to evaluate

    Returns:
        fig: matplotlib Figure
        calibration_stats: dict with calibration statistics
    """
    from scipy import stats as scipy_stats

    # Flatten arrays
    truths_flat = all_truths.flatten()
    preds_flat = all_preds.flatten()
    sigmas_flat = all_sigmas.flatten()

    # Remove NaN and invalid sigma values
    valid_mask = (
        ~np.isnan(truths_flat) &
        ~np.isnan(preds_flat) &
        ~np.isnan(sigmas_flat) &
        (sigmas_flat > 0)
    )
    truths_valid = truths_flat[valid_mask]
    preds_valid = preds_flat[valid_mask]
    sigmas_valid = sigmas_flat[valid_mask]

    n_valid = len(truths_valid)

    # Define confidence levels to evaluate (e.g., 10%, 20%, ..., 99%)
    confidence_levels = np.linspace(0.05, 0.99, n_bins)

    # For each confidence level, compute the observed coverage
    observed_coverages = []
    for conf in confidence_levels:
        # For a Gaussian, the central conf% interval is [μ - z*σ, μ + z*σ]
        # where z = Φ^(-1)((1 + conf) / 2)
        z_score = scipy_stats.norm.ppf((1 + conf) / 2)

        lower = preds_valid - z_score * sigmas_valid
        upper = preds_valid + z_score * sigmas_valid

        # Fraction of truths within this interval
        within_interval = (truths_valid >= lower) & (truths_valid <= upper)
        observed_coverage = np.mean(within_interval)
        observed_coverages.append(observed_coverage)

    observed_coverages = np.array(observed_coverages)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot reliability curve
    ax.plot(confidence_levels, observed_coverages, 'o-', color='steelblue',
            linewidth=2, markersize=6, label='Observed coverage')

    # Plot reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')

    # Shade regions
    ax.fill_between(confidence_levels, confidence_levels, observed_coverages,
                    where=(observed_coverages < confidence_levels),
                    alpha=0.3, color='orange', label='Overconfident')
    ax.fill_between(confidence_levels, confidence_levels, observed_coverages,
                    where=(observed_coverages > confidence_levels),
                    alpha=0.3, color='green', label='Underconfident')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted Confidence Level')
    ax.set_ylabel('Observed Coverage (fraction within interval)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Compute summary statistics
    # Mean absolute calibration error
    mace = np.mean(np.abs(observed_coverages - confidence_levels))

    # Root mean square calibration error
    rmsce = np.sqrt(np.mean((observed_coverages - confidence_levels) ** 2))

    # Specific coverage checks
    idx_50 = np.argmin(np.abs(confidence_levels - 0.50))
    idx_90 = np.argmin(np.abs(confidence_levels - 0.90))
    idx_95 = np.argmin(np.abs(confidence_levels - 0.95))

    coverage_at_50 = observed_coverages[idx_50]
    coverage_at_90 = observed_coverages[idx_90]
    coverage_at_95 = observed_coverages[idx_95]

    calibration_stats = {
        'mace': mace,
        'rmsce': rmsce,
        'coverage_at_50': coverage_at_50,
        'coverage_at_90': coverage_at_90,
        'coverage_at_95': coverage_at_95,
        'confidence_levels': confidence_levels,
        'observed_coverages': observed_coverages,
        'n_valid': n_valid
    }

    # Add statistics text box
    stats_text = (
        f'n = {n_valid:,}\n'
        f'MACE: {mace:.3f}\n'
        f'RMSCE: {rmsce:.3f}\n'
        f'At 50% conf: {coverage_at_50:.1%} observed\n'
        f'At 90% conf: {coverage_at_90:.1%} observed\n'
        f'At 95% conf: {coverage_at_95:.1%} observed'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    _save_fig(fig, save_path)

    return fig, calibration_stats


def _plot_training_curves_single(
    fold_stats: pd.DataFrame,
    x_max: int,
    title_suffix: str,
    save_path: str | Path | None = None,
    log_corr: bool = False,
):
    """Plot a single two-panel training curve figure (correlations + error/loss).

    Returns:
        matplotlib Figure.
    """
    palette = sns.color_palette()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Correlations
    ax = axes[0]
    long = fold_stats.melt(
        id_vars='Epoch',
        value_vars=['Pearson correlation', 'Spearman correlation'],
        var_name='Metric',
        value_name='Correlation'
    )
    sns.lineplot(
        data=long, x='Epoch', y='Correlation', hue='Metric',
        palette={'Pearson correlation': palette[0], 'Spearman correlation': palette[1]},
        ax=ax
    )
    ax.set_title(f'Correlation Scores ({title_suffix})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Correlation Score')
    if log_corr:
        ax.set_yscale('symlog', linthresh=0.01)
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, x_max)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Metric', loc='lower right')

    # Error / Loss
    ax = axes[1]
    long = fold_stats.melt(
        id_vars='Epoch',
        value_vars=['Mean absolute error', 'train NLL', 'test NLL'],
        var_name='Error / Loss',
        value_name='Value'
    )
    sns.lineplot(
        data=long, x='Epoch', y='Value', hue='Error / Loss',
        palette={'Mean absolute error': palette[2], 'train NLL': palette[3], 'test NLL': palette[4]},
        ax=ax
    )
    ax.set_title(f'Error/Loss ({title_suffix})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error / Loss')
    ax.set_xlim(0, x_max)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Metric', loc='upper right')

    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


def plot_training_curves(
    stats: pd.DataFrame,
    save_path: str | Path | None = None,
    log_corr: bool = False,
):
    """Plot training curves: one figure per fold plus one for the fold average.

    Args:
        stats: DataFrame with columns Fold, Epoch, Pearson correlation,
               Spearman correlation, Mean absolute error, train NLL, test NLL.
        save_path: Optional path prefix. Each figure is saved by appending
               ``_fold{N}.png`` or ``_all.png`` to this prefix.
        log_corr: If True, use a symmetric-log scale on the correlation y-axis.

    Returns:
        List of matplotlib Figures (one per fold, then the average).
    """
    sns.set_theme(context="paper", style="whitegrid")

    folds = sorted(stats['Fold'].unique())
    x_max = stats['Epoch'].max()

    if save_path is not None:
        save_path = Path(save_path)
        # Strip extension so callers can pass e.g. "plots/training.png"
        suffix = save_path.suffix
        stem = save_path.with_suffix('')

    figures: list[plt.Figure] = []

    for fold in folds:
        fold_save = None
        if save_path is not None:
            fold_save = f"{stem}_fold{fold}{suffix}"
        fig = _plot_training_curves_single(
            stats[stats['Fold'] == fold], x_max, f'Fold {fold}', fold_save,
            log_corr=log_corr,
        )
        figures.append(fig)

    # Average across folds
    avg_stats = stats.groupby('Epoch').mean(numeric_only=True).reset_index()
    avg_save = None
    if save_path is not None:
        avg_save = f"{stem}_all{suffix}"
    fig = _plot_training_curves_single(
        avg_stats, x_max, 'Average across folds', avg_save,
        log_corr=log_corr,
    )
    figures.append(fig)

    return figures

def plot_attention_maps(
    model: nn.Module,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_target_static: torch.Tensor,
    day_idx: int,
    target_point_idx: int,
    grid_shape: tuple[int, int],
    metadata: ds.Era5Metadata,
    device: torch.device,
    y_target_flat: np.ndarray | None = None,
    show_average: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot Laplace, mean-attribute, and variance attention weights for one target point.
    First row: single day (day_idx). Second row: weights averaged over all days.

    Args:
        model:             Trained SMACNP model.
        x_context:         (T, N, 6) context attributes.
        y_context:         (T, N, 1) context observations.
        x_target_static:   (M, 4) static target attributes [lat, lon, alt, mTPI].
        day_idx:           Time index to visualize in the first row.
        target_point_idx:  Row index into x_target_static (which target point to inspect).
        grid_shape:        (N, E) shape of the MeteoSwiss target grid.
        metadata:          Era5Metadata with lat/lon bounds and denormalize method.
        device:            Torch device.
        y_target_flat:     Optional (N*E,) ground-truth array to derive the Switzerland mask.
        show_average:      If True, add a second row with weights averaged over all days.
        save_path:         Optional path to save the figure.
    """
    model.eval()

    # Denormalize coordinates
    lat_range = metadata.lat_max - metadata.lat_min
    lon_range = metadata.lon_max - metadata.lon_min
    target_lats = x_target_static[:, 0].cpu().numpy() * lat_range + metadata.lat_min
    target_lons = x_target_static[:, 1].cpu().numpy() * lon_range + metadata.lon_min
    tgt_lat = target_lats[target_point_idx]
    tgt_lon = target_lons[target_point_idx]

    static_pt = x_target_static[target_point_idx:target_point_idx + 1].to(device)

    def _forward_day(d):
        xc = x_context[d:d + 1].to(device)
        yc = y_context[d:d + 1].to(device)
        cos_doy = xc[0, 0, 4].item()
        sin_doy = xc[0, 0, 5].item()
        seasonal = torch.tensor([[cos_doy, sin_doy]], device=device)
        xt = torch.cat([static_pt, seasonal], dim=-1).unsqueeze(0)
        with torch.no_grad():
            _ = model(xc, yc, xt)
        lw = model.mean_loc_encoder.laplace_attention.last_weights[0, 0].cpu().numpy()
        mw = model.mean_attr_encoder.cross_attention.last_weights[0, 0].cpu().numpy()
        vw = model.variance_encoder.cross_attention.last_weights[0, 0].cpu().numpy()
        return lw, mw, vw

    # Single day
    laplace_w, mean_w, var_w = _forward_day(day_idx)

    # Average over all days
    if show_average:
        T = x_context.shape[0]
        laplace_sum = np.zeros_like(laplace_w)
        mean_sum    = np.zeros_like(mean_w)
        var_sum     = np.zeros_like(var_w)
        for d in range(T):
            lw, mw, vw = _forward_day(d)
            laplace_sum += lw
            mean_sum    += mw
            var_sum     += vw
        laplace_avg = laplace_sum / T
        mean_avg    = mean_sum    / T
        var_avg     = var_sum     / T

    # ERA5 grid shape from unique lat/lon values
    context_lats = x_context[0, :, 0].cpu().numpy() * lat_range + metadata.lat_min
    context_lons = x_context[0, :, 1].cpu().numpy() * lon_range + metadata.lon_min
    n_era5_lats = len(np.unique(np.round(context_lats, 6)))
    n_era5_lons = len(np.unique(np.round(context_lons, 6)))

    # Upsample + mask helpers
    N, E = grid_shape
    zoom_factors = (N / n_era5_lats, E / n_era5_lons)
    swiss_mask = ~np.isnan(y_target_flat.reshape(N, E)) if y_target_flat is not None else np.ones((N, E), dtype=bool)
    extent = [metadata.lon_min, metadata.lon_max, metadata.lat_min, metadata.lat_max]
    center_lat = (metadata.lat_min + metadata.lat_max) / 2
    geo_aspect = 1.0 / np.cos(np.radians(center_lat))

    def _prepare(w):
        w_2d = w.reshape(n_era5_lats, n_era5_lons)
        w_up = zoom(w_2d, zoom_factors, order=1)
        w_up = np.flipud(w_up)
        return np.where(swiss_mask, w_up, np.nan)

    def _plot_row(axes_row, weights, row_label):
        titles = ['Laplace (spatial)', 'Mean-attribute', 'Variance']
        for ax, w, title in zip(axes_row, weights, titles):
            im = ax.imshow(_prepare(w), cmap='YlOrRd', origin='lower', vmin=0,
                           extent=extent, aspect=geo_aspect)
            ax.scatter([tgt_lon], [tgt_lat], marker='*', c='blue', s=250,
                       zorder=5, label='Target point')
            plt.colorbar(im, ax=ax, label='Attention weight')
            ax.set_title(f'{title}\n{row_label}')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.legend()

    n_rows = 2 if show_average else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # make indexing uniform

    _plot_row(axes[0], [laplace_w, mean_w, var_w], f'Day {day_idx}')
    if show_average:
        _plot_row(axes[1], [laplace_avg, mean_avg, var_avg], f'Average over {T} days')

    plt.suptitle(
        f"Attention weights — target #{target_point_idx} ({tgt_lat:.3f}°N, {tgt_lon:.3f}°E)",
        y=1.02,
    )
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def find_target_point(
    lat: float,
    lon: float,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> tuple[int, float, float, float]:
    """
    Find the target point index closest to a given (lat, lon) coordinate.

    Args:
        lat:         Desired latitude in degrees.
        lon:         Desired longitude in degrees.
        target_lats: (M,) array of target point latitudes.
        target_lons: (M,) array of target point longitudes.

    Returns:
        Tuple of (index, matched_lat, matched_lon, distance_km).
    """
    dlat = target_lats - lat
    dlon = (target_lons - lon) * np.cos(np.radians(lat))   # scale lon by cos(lat)
    dist_deg = np.sqrt(dlat**2 + dlon**2)
    idx = int(np.argmin(dist_deg))
    dist_km = dist_deg[idx] * 111.0   # 1° ≈ 111 km
    return idx, float(target_lats[idx]), float(target_lons[idx]), dist_km

def get_lat_lon_arrays(
    x_context: torch.Tensor,
    x_target_static: torch.Tensor,
    metadata: ds.Era5Metadata,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Denormalize lat/lon coordinates back to degrees for context and target points.

    Args:
        x_context:        (T, N, 6) context attributes from build_smacnp_context.
        x_target_static:  (M, 4) static target attributes from build_smacnp_targets.
        metadata:         Era5Metadata with lat/lon bounds used for normalization.

    Returns:
        context_lats, context_lons, target_lats, target_lons — all (N,) or (M,) in degrees.
    """
    lat_range = metadata.lat_max - metadata.lat_min
    lon_range = metadata.lon_max - metadata.lon_min

    context_lats = x_context[0, :, 0].cpu().numpy() * lat_range + metadata.lat_min
    context_lons = x_context[0, :, 1].cpu().numpy() * lon_range + metadata.lon_min

    target_lats = x_target_static[:, 0].cpu().numpy() * lat_range + metadata.lat_min
    target_lons = x_target_static[:, 1].cpu().numpy() * lon_range + metadata.lon_min

    return context_lats, context_lons, target_lats, target_lons

