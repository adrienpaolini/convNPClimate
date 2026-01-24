"""Visualization utilities for convCNP prediction analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from scipy.ndimage import zoom

import datasets as ds


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
    residual_range: tuple[float, float] = (-10, 10)
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
    era5_upsampled = zoom(era5_input, zoom_factors, order=0)  # bilinear interpolation
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
        truth_grid = truth_grid - 273.15
        pred_grid = pred_grid - 273.15
        era5_upsampled = era5_upsampled - 273.15
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
    axes[0].set_title(f'ERA5 Input ({unit})')
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

    return fig


def plot_error_maps(
    mae_grid: np.ndarray,
    rmse_grid: np.ndarray,
    bias_grid: np.ndarray,
    title: str,
    error_range: tuple[float, float] = (0, 5),
    bias_range: tuple[float, float] = (-3, 3)
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

    return fig


def plot_uncertainty_vs_error(
    sigma_grid: np.ndarray,
    abs_error_grid: np.ndarray,
    title: str,
    sigma_range: tuple[float, float] = (0, 5),
    error_range: tuple[float, float] = (0, 10)
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

    return fig


def plot_crps_map(
    crps_grid: np.ndarray,
    title: str,
    crps_range: tuple[float, float] = (0, 3)
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

    return fig, mean_crps


def plot_skill_map(
    skill_grid: np.ndarray,
    global_skill: float,
    title: str,
    skill_range: tuple[float, float] = (-1, 1)
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

    return fig


def plot_perpixel_uncertainty_vs_error(
    mean_sigma_grid: np.ndarray,
    mae_grid: np.ndarray,
    title: str,
    sigma_range: tuple[float, float] = (0, 5),
    error_range: tuple[float, float] = (0, 5)
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
    y_cmap: str = 'YlOrRd'
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

    return fig, correlation


def plot_temporal_error_analysis(
    all_errors: np.ndarray,
    holdout_dates: np.ndarray,
    title: str
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

    return fig, correlation


def plot_training_curves(stats: pd.DataFrame):
    """Plot per-fold training curves: correlations and error/loss.

    Args:
        stats: DataFrame with columns Fold, Epoch, Pearson correlation,
               Spearman correlation, Mean absolute error, train NLL, test NLL.

    Returns:
        matplotlib Figure.
    """
    sns.set_theme(context="paper", style="whitegrid")

    folds = sorted(stats['Fold'].unique())
    n_folds = len(folds)

    fig, axes = plt.subplots(nrows=n_folds, ncols=2, figsize=(12, 12), sharex=False)
    if n_folds == 1:
        axes = axes.reshape(1, -1)

    for i, fold in enumerate(folds):
        fold_stats = stats[stats['Fold'] == fold]
        palette = sns.color_palette()

        # Correlations
        ax = axes[i, 0]
        fold_long = fold_stats.melt(
            id_vars='Epoch',
            value_vars=['Pearson correlation', 'Spearman correlation'],
            var_name='Metric',
            value_name='Correlation'
        )
        sns.lineplot(
            data=fold_long, x='Epoch', y='Correlation', hue='Metric',
            palette={'Pearson correlation': palette[0], 'Spearman correlation': palette[1]},
            ax=ax
        )
        ax.set_title(f'Correlation Scores (Fold {fold})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Correlation Score')
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, stats['Epoch'].max())
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='Metric', loc='lower right')

        # Error / Loss
        ax = axes[i, 1]
        fold_long = fold_stats.melt(
            id_vars='Epoch',
            value_vars=['Mean absolute error', 'train NLL', 'test NLL'],
            var_name='Error / Loss',
            value_name='Value'
        )
        sns.lineplot(
            data=fold_long, x='Epoch', y='Value', hue='Error / Loss',
            palette={'Mean absolute error': palette[2], 'train NLL': palette[3], 'test NLL': palette[4]},
            ax=ax
        )
        ax.set_title(f'Error/Loss (Fold {fold})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error / Loss')
        ax.set_xlim(0, stats['Epoch'].max())
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='Metric', loc='upper right')

    plt.tight_layout()
    return fig
