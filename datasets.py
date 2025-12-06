"""
Dataset loading and metadata utilities for ERA5 and MeteoSwiss data.
"""

import dataclasses
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import xarray as xr

from json_utils import save_dataclass_json, load_dataclass_json
from convCNP.validation.utils import get_dists


@dataclasses.dataclass
class Era5Metadata:
    """
    Holds metadata and statistics for input data.

    These are required:
     - to normalize target coordinates into the same reference
       frame as the input context.
     - to calculate distances of to points in the target set
    """
    data_mean: float
    data_std: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    # Original coordinate arrays from the input grid (for distance calculations)
    lat_coords: np.ndarray  # Shape: (n_lat,)
    lon_coords: np.ndarray  # Shape: (n_lon,)

    def denormalize(self, normalized_data: float | np.ndarray) -> float | np.ndarray:
        """
        Converts normalized data back to original units (Kelvin).
        Formula: Original = (Normalized * Std) + Mean
        """
        return (normalized_data * self.data_std) + self.data_mean


# Fields containing numpy arrays that need special JSON handling
ERA5_METADATA_NUMPY_FIELDS = ['lat_coords', 'lon_coords']


def save_metadata_json(metadata: Era5Metadata, output_path: Path) -> None:
    """Save Era5Metadata dataclass as JSON to specified path."""
    save_dataclass_json(metadata, output_path, numpy_fields=ERA5_METADATA_NUMPY_FIELDS)


def load_metadata_json(metadata_json: Path) -> Era5Metadata:
    """Load Era5Metadata dataclass from JSON file at specified path."""
    return load_dataclass_json(Era5Metadata, metadata_json, numpy_fields=ERA5_METADATA_NUMPY_FIELDS)


def load_era5_data(
    var_glob: str,
    var_name: str = 't2m_max',
    year_start: int | None = None,
    device: torch.device | None = None
) -> Tuple[torch.Tensor, Era5Metadata]:
    """
    Loads ERA5 data from NetCDF files matching the given glob pattern.
    Optionally filters data starting from a specific year.

    Args:
        var_glob: Glob pattern for NetCDF files
        var_name: Name of the variable in the dataset
        year_start: Optional year to filter data from
        device: Torch device to load tensor to (defaults to CPU)

    Returns:
        tensor: Processed input tensor with shape (time, channels, lat, lon)
        stats: Era5Metadata containing normalization parameters and grid coordinates
    """
    if device is None:
        device = torch.device('cpu')

    # Data load
    print(f"Loading ERA5 data from: {var_glob}")
    ds = xr.open_mfdataset(var_glob, combine='by_coords')
    print(f"Dataset loaded with dimensions: {ds.dims}")
    print(f"Original dataset time range: {ds.time.min().values} to {ds.time.max().values}")

    # Data filtering
    if year_start is not None:
        ds = ds.sel(time=slice(f'{year_start}-01-01', None))
        print(f"Filtered dataset time range from {year_start}: {ds.time.min().values} to {ds.time.max().values}")
    print(f"Filtered dataset dimensions: {ds.dims}")

    # Normalization
    # Note: we explicitly call .compute() here because dask lazy arrays cannot
    # be converted to python scalars via .item() directly.
    print(f"Calculating stats for {var_name} (Kelvin)...")
    mean_val = ds[var_name].mean().compute().item()
    std_val = ds[var_name].std().compute().item()
    lat_min = ds.latitude.min().compute().item()
    lat_max = ds.latitude.max().compute().item()
    lon_min = ds.longitude.min().compute().item()
    lon_max = ds.longitude.max().compute().item()

    # Extract coordinate arrays for distance calculations
    lat_coords = ds.latitude.values
    lon_coords = ds.longitude.values

    stats = Era5Metadata(
        data_mean=mean_val,
        data_std=std_val,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_coords=lat_coords,
        lon_coords=lon_coords,
    )
    norm_data = (ds[var_name] - mean_val) / std_val

    # Normalize coordinates
    lat_norm = (ds.latitude - lat_min) / (lat_max - lat_min)
    lon_norm = (ds.longitude - lon_min) / (lon_max - lon_min)
    # Xarray broadcast automatically expands (lat) and (lon) to (time, lat, lon)
    # dependent on the shape of `norm_data`.
    lat_channel, lon_channel, _ = xr.broadcast(lat_norm, lon_norm, norm_data)

    # Time Embeddings
    day_of_year = ds.time.dt.dayofyear
    time_rads = (day_of_year - 1) / 365.0 * 2 * np.pi
    # Use numpy functions; if inputs are dask/xarray, result is lazy
    cos_time = np.cos(time_rads)
    sin_time = np.sin(time_rads)
    # Broadcast time to spatial dimensions
    cos_channel, sin_channel, _ = xr.broadcast(cos_time, sin_time, norm_data)

    # Channel stacking
    # Create the list of channels
    channel_list = [norm_data, lat_channel, lon_channel, cos_channel, sin_channel]
    channel_names = ['data', 'lat', 'lon', 'cos_time', 'sin_time']

    # xr.concat preserves dask arrays.
    tensor_Z = xr.concat(channel_list, dim="channel")

    # Assign labels to the channel dimension so we know what is what
    tensor_Z = tensor_Z.assign_coords(channel=channel_names)

    # Order matches: [Data, Lat, Lon, CosTime, SinTime]
    tensor_Z = tensor_Z.transpose('time', 'channel', 'latitude', 'longitude')
    # Use float32 consistently - the model and mask generation use float32
    tensor_Z = torch.from_numpy(tensor_Z.values.astype(np.float32)).to(device)
    print(f"Final tensor_Z torch tensor with shape (time, channel, lat, lon): {tensor_Z.shape}, dtype: {tensor_Z.dtype}")

    return tensor_Z, stats


def prepare_meteoswiss_targets(
    meteo_swiss_glob: str,
    normalization_stats: Era5Metadata,
    data_var: str = 'TmaxD',
    elev_var: str | None = None,
    convert_to_kelvin: bool = False,
    year_start: int | None = None
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Transforms the MeteoSwiss Dataset into the Target Tensors (x, y, e).

    Args:
        meteo_swiss_glob: Glob pattern for MeteoSwiss NetCDF files
        normalization_stats: Stats from input for normalization
        data_var: Name of temperature variable (default 'TmaxD')
        elev_var: Name of elevation variable (if exists)
        convert_to_kelvin: Set True if MeteoSwiss is Celsius and input is Kelvin
        year_start: Optional year to filter data from

    Returns:
        X (Locations): (point, 2) -> [Lat_norm, Lon_norm]
        Y (Truth):     (time, point) -> [Temp_norm]
        E (Topo):      (point, 3) -> [True_Elev, Elev_Diff, mTPI] (Zeros if missing)
    """
    ds = xr.open_mfdataset(meteo_swiss_glob, combine='by_coords', data_vars='all')
    if year_start is not None:
        ds = ds.sel(time=slice(f'{year_start}-01-01', None))

    print(f"--- Preparing MeteoSwiss Targets (x, y, e) ---")

    # 1. Flatten the Grid
    # Your dataset has dims (time, N, E).
    # We stack N and E to create a list of points.
    print(f"Flattening ({ds.sizes['N']}, {ds.sizes['E']}) grid to (point)...")

    # This operation is lazy in xarray/dask.
    # It creates a MultiIndex, but doesn't move data yet.
    ds_flat = ds.stack(point=("N", "E"))

    # 2. Prepare Target Locations (Tensor x)
    print("Creating Target Coordinate Tensor (x)...")

    # Extract the 2D lat/lon (now flattened to 1D)
    # Note: dask arrays remain lazy.
    lat_flat = ds_flat['lat']
    lon_flat = ds_flat['lon']

    # === COORDINATE VERIFICATION ===
    # Check that MeteoSwiss lat/lon are in degrees (WGS84), not LV95 meters
    lat_sample = lat_flat.values[:5] if hasattr(lat_flat.values, '__len__') else lat_flat.compute().values[:5]
    lon_sample = lon_flat.values[:5] if hasattr(lon_flat.values, '__len__') else lon_flat.compute().values[:5]

    print(f"\n=== COORDINATE VERIFICATION ===")
    print(f"MeteoSwiss lat (first 5 points): {lat_sample}")
    print(f"MeteoSwiss lon (first 5 points): {lon_sample}")
    print(f"MeteoSwiss lat range: [{float(lat_flat.min().compute()):.4f}, {float(lat_flat.max().compute()):.4f}]")
    print(f"MeteoSwiss lon range: [{float(lon_flat.min().compute()):.4f}, {float(lon_flat.max().compute()):.4f}]")
    print(f"\nERA5 normalization bounds (should overlap with MeteoSwiss if both in degrees):")
    print(f"ERA5 lat range: [{normalization_stats.lat_min:.4f}, {normalization_stats.lat_max:.4f}]")
    print(f"ERA5 lon range: [{normalization_stats.lon_min:.4f}, {normalization_stats.lon_max:.4f}]")

    # Sanity check: if MeteoSwiss coords are in LV95 meters, they'll be ~1e6, not ~45-48 degrees
    if lat_sample.mean() > 1000:
        print(f"\n*** WARNING: MeteoSwiss lat values ({lat_sample.mean():.0f}) look like LV95 meters, not WGS84 degrees! ***")
        print(f"*** Expected lat in range ~45-48 for Switzerland ***")
    if lon_sample.mean() > 1000:
        print(f"\n*** WARNING: MeteoSwiss lon values ({lon_sample.mean():.0f}) look like LV95 meters, not WGS84 degrees! ***")
        print(f"*** Expected lon in range ~5-11 for Switzerland ***")
    print(f"=== END COORDINATE VERIFICATION ===\n")

    # Normalize using input boundaries
    lat_norm = (lat_flat - normalization_stats.lat_min) / (normalization_stats.lat_max - normalization_stats.lat_min)
    lon_norm = (lon_flat - normalization_stats.lon_min) / (normalization_stats.lon_max - normalization_stats.lon_min)

    # Stack into (point, 2)
    tensor_x = xr.concat([lat_norm, lon_norm], dim="coord")
    tensor_x = tensor_x.transpose("point", "coord")
    tensor_x = tensor_x.assign_coords(coord=["lat", "lon"])

    # 3. Prepare Ground Truth Data (Tensor y)
    print(f"Creating Ground Truth Tensor (y) for {data_var}...")

    data_flat = ds_flat[data_var]

    # --- UNIT CONVERSION CHECK ---
    if convert_to_kelvin:
        print("  -> Converting Celsius to Kelvin (+273.15) before normalization")
        data_flat = data_flat + 273.15

    # Normalize using stats (Z-Score)
    # Truth = (Temp - Input_Mean) / Input_Std
    y_norm = (data_flat - normalization_stats.data_mean) / normalization_stats.data_std

    tensor_y = y_norm.transpose("time", "point")

    # 4. Prepare Topography (Tensor e)
    # The paper expects 3 features: [True Elevation, Elevation Diff, mTPI]
    print("Creating Elevation Tensor (e) [3 features]...")

    feature_names = ['true_elev', 'elev_diff', 'mTPI']

    if elev_var and elev_var in ds_flat:
        print(f"  -> Found elevation '{elev_var}'. Filling feature 0, zeroing 1 & 2.")
        elev_flat = ds_flat[elev_var]

        # Create zeros for the missing features (Diff and mTPI)
        # xr.zeros_like creates a lazy dask array if input is dask
        zeros = xr.zeros_like(elev_flat)

        # Stack: [Elev, 0, 0]
        tensor_e = xr.concat([elev_flat, zeros, zeros], dim="feature")
    else:
        print("  -> Warning: No elevation variable found. Filling ALL 3 features with Zeros.")
        # Create zeros matching the spatial dimension
        zeros = xr.zeros_like(lat_flat)

        # Stack: [0, 0, 0]
        tensor_e = xr.concat([zeros, zeros, zeros], dim="feature")

    # Assign coordinates and transpose to (point, feature)
    tensor_e = tensor_e.assign_coords(feature=feature_names)
    tensor_e = tensor_e.transpose("point", "feature")

    print(f"Final X (Locations): {tensor_x.sizes}")
    print(f"Final Y (Targets):   {tensor_y.sizes}")
    print(f"Final E (Topo):      {tensor_e.sizes}")

    return tensor_x, tensor_y, tensor_e


def calculate_dists_meteoswiss(
    normalization_stats: Era5Metadata,
    meteoswiss_targets: xr.DataArray,
    device: torch.device | None = None
) -> torch.Tensor:
    """
    Get the distances between the ERA5 grid points and MeteoSwiss target points.

    Args:
        normalization_stats: Stats containing ERA5 grid coordinates and normalization bounds
        meteoswiss_targets: xarray.DataArray with normalized target coordinates (target_x)
        device: Torch device to load tensor to (defaults to CPU)

    Returns:
        Tensor of shape (n_target_points, n_lat, n_lon) with squared distances
    """
    if device is None:
        device = torch.device('cpu')

    # Build ERA5 grid from stored coordinates
    era5_lon_grid, era5_lat_grid = np.meshgrid(
        normalization_stats.lon_coords,
        normalization_stats.lat_coords
    )
    era5_lat_grid = torch.from_numpy(era5_lat_grid).float().to(device)
    era5_lon_grid = torch.from_numpy(era5_lon_grid).float().to(device)

    # De-normalize MeteoSwiss coordinates back to degrees
    lat_norm = meteoswiss_targets.sel(coord='lat').values
    lon_norm = meteoswiss_targets.sel(coord='lon').values

    lat = lat_norm * (normalization_stats.lat_max - normalization_stats.lat_min) + normalization_stats.lat_min
    lon = lon_norm * (normalization_stats.lon_max - normalization_stats.lon_min) + normalization_stats.lon_min

    # Combine into coordinate array for get_dists
    meteoswiss_coords = np.stack([lat, lon], axis=1)

    # get_dists returns a CPU tensor, move to requested device
    dists = get_dists(meteoswiss_coords, era5_lat_grid, era5_lon_grid)
    return dists.to(device)


def get_meteoswiss_grid_shape(meteo_swiss_glob: str) -> Tuple[int, int]:
    """
    Get the original grid shape (N, E) from MeteoSwiss data.

    Args:
        meteo_swiss_glob: Glob pattern for MeteoSwiss NetCDF files

    Returns:
        Tuple of (N, E) grid dimensions
    """
    ds = xr.open_mfdataset(meteo_swiss_glob, combine='by_coords', data_vars='all')
    return ds.sizes['N'], ds.sizes['E']
