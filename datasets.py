"""
Dataset loading and metadata utilities for ERA5 and MeteoSwiss data.
"""

import dataclasses
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pyproj
import torch
import torch.nn.functional as F
import xarray as xr

from json_utils import save_dataclass_json, load_dataclass_json
from convCNP.validation.utils import get_dists


KELVIN_OFFSET = 273.15

# Coordinate transformers for Swiss LV95 (EPSG:2056) <-> WGS84 (EPSG:4326)
_WGS84_TO_LV95 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
_LV95_TO_WGS84 = pyproj.Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)


def wgs84_to_lv95(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert WGS84 (lon, lat) to Swiss LV95 (x, y) coordinates."""
    x, y = _WGS84_TO_LV95.transform(lon, lat)
    return x, y


def lv95_to_wgs84(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Swiss LV95 (x, y) to WGS84 (lon, lat) coordinates."""
    lon, lat = _LV95_TO_WGS84.transform(x, y)
    return lon, lat


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


@dataclasses.dataclass
class DataPaths:
    """Glob patterns and paths for all datasets."""
    ERA5_MAX_TEMP_GLOB: str
    ERA5_PRECIP_GLOB: str
    ERA5_GEOPOTENTIAL_GLOB: str
    EOBS_MAX_TEMP_GLOB: str
    EOBS_PRECIP_GLOB: str
    METEO_SWISS_MAX_TEMP_GLOB: str
    METEO_SWISS_PRECIP_GLOB: str
    HI_RES_TOPOGRAPHY_ZARR_PATH: str


def build_data_paths(base_dataset_dir: Path | str) -> DataPaths:
    """Construct all dataset glob patterns from a base directory.

    Args:
        base_dataset_dir: Root directory containing the dataset subdirectories.

    Returns:
        DataPaths with all glob patterns resolved.
    """
    base = Path(base_dataset_dir)
    return DataPaths(
        ERA5_MAX_TEMP_GLOB=str(base / 'ERA5_Land/max_temperature') + '/*.nc',
        ERA5_PRECIP_GLOB=str(base / 'ERA5_Land/precipitation') + '/*.nc',
        ERA5_GEOPOTENTIAL_GLOB=str(base / 'ERA5_Land/geopotential') + '/*.nc',
        EOBS_MAX_TEMP_GLOB=str(base / 'EOBS/max_temperature') + '/*.nc',
        EOBS_PRECIP_GLOB=str(base / 'EOBS/precipitation') + '/*.nc',
        METEO_SWISS_MAX_TEMP_GLOB=str(base / 'MeteoSwiss/TmaxD_v2.0_swiss.lv95') + '/*.nc',
        METEO_SWISS_PRECIP_GLOB=str(base / 'MeteoSwiss/RhiresD_v2.0_swiss.lv95') + '/*.nc',
        HI_RES_TOPOGRAPHY_ZARR_PATH=str(base / 'topo_subset.zarr'),
    )


def load_era5_data(
    var_glob: Path | str,
    var_name: str = 't2m_max',
    year_start: int | None = None,
    geopotential_glob: Path | str | None = None,
    geopotential_var_name: str = 'z',
    include_seasonal_embeddings = True,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, Era5Metadata, torch.Tensor | None, np.ndarray]:
    """
    Loads ERA5 data from NetCDF files matching the given glob pattern.
    Optionally filters data starting from a specific year.
    Optionally loads geopotential data and converts to altitude in meters.

    Args:
        var_glob: Glob pattern for NetCDF files
        var_name: Name of the variable in the dataset
        year_start: Optional year to filter data from
        geopotential_glob: Optional glob pattern for geopotential NetCDF files
        geopotential_var_name: Name of geopotential variable (default 'z')
        device: Torch device to load tensor to (defaults to CPU)

    Returns:
        tensor: Processed input tensor with shape (time, channels, lat, lon)
        stats: Era5Metadata containing normalization parameters and grid coordinates
        altitude: Tensor with altitude in meters, shape (lat, lon), or None if geopotential_glob not provided
        time_coords: Array of datetime64 values corresponding to each time index
    """
    if device is None:
        device = torch.device('cpu')

    var_glob = str(var_glob)  # Ensure it's a string for xarray
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
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    if std_val == 0:
        raise ValueError(f"ERA5 variable '{var_name}' has zero standard deviation (constant field); cannot normalize.")
    if lat_range == 0 or lon_range == 0:
        raise ValueError(f"ERA5 coordinate range is zero (lat_range={lat_range}, lon_range={lon_range}); cannot normalize.")
    norm_data = (ds[var_name] - mean_val) / std_val

    # Normalize coordinates
    lat_norm = (ds.latitude - lat_min) / lat_range
    lon_norm = (ds.longitude - lon_min) / lon_range
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
    channel_list = [norm_data, lat_channel, lon_channel]
    channel_names = ['data', 'lat', 'lon']

    if include_seasonal_embeddings:
        channel_list.extend([cos_channel, sin_channel])
        channel_names.extend(['cos_time', 'sin_time'])

    # xr.concat preserves dask arrays.
    tensor_Z = xr.concat(channel_list, dim="channel")

    # Assign labels to the channel dimension so we know what is what
    tensor_Z = tensor_Z.assign_coords(channel=channel_names)

    # Order matches: [Data, Lat, Lon, CosTime, SinTime]
    tensor_Z = tensor_Z.transpose('time', 'channel', 'latitude', 'longitude')

    # Extract time coordinates before converting to tensor
    time_coords = ds.time.values

    # Use float32 consistently - the model and mask generation use float32
    tensor_Z = torch.from_numpy(tensor_Z.values.astype(np.float32)).to(device)
    print(f"Final tensor_Z torch tensor with shape (time, channel, lat, lon): {tensor_Z.shape}, dtype: {tensor_Z.dtype}")

    # Load geopotential and convert to altitude if provided
    altitude_tensor, altitude = None, None
    if geopotential_glob is not None:
        geopotential_glob = str(geopotential_glob)
        print(f"\nLoading geopotential data from: {geopotential_glob}")
        geo_ds = xr.open_mfdataset(geopotential_glob, combine='by_coords')
        print(f"Geopotential dataset dimensions: {geo_ds.dims}")

        # Verify grids match
        geo_lat = geo_ds.latitude.values
        geo_lon = geo_ds.longitude.values
        print(f"Geopotential lat range: [{geo_lat.min():.4f}, {geo_lat.max():.4f}], shape: {geo_lat.shape}")
        print(f"Geopotential lon range: [{geo_lon.min():.4f}, {geo_lon.max():.4f}], shape: {geo_lon.shape}")
        print(f"ERA5 data lat range: [{lat_coords.min():.4f}, {lat_coords.max():.4f}], shape: {lat_coords.shape}")
        print(f"ERA5 data lon range: [{lon_coords.min():.4f}, {lon_coords.max():.4f}], shape: {lon_coords.shape}")

        # Check if grids match exactly
        grids_match_exactly = (
            geo_lat.shape == lat_coords.shape and
            geo_lon.shape == lon_coords.shape and
            np.allclose(geo_lat, lat_coords, atol=1e-6) and
            np.allclose(geo_lon, lon_coords, atol=1e-6)
        )

        if grids_match_exactly:
            print("✓ Geopotential and ERA5 data grids match exactly")
        else:
            # Check if ERA5 coords are a subset of geopotential coords (within tolerance)
            # This handles cases where geopotential covers a larger area but same resolution
            lat_subset = all(np.min(np.abs(geo_lat - t)) < 1e-4 for t in lat_coords)
            lon_subset = all(np.min(np.abs(geo_lon - t)) < 1e-4 for t in lon_coords)

            if lat_subset and lon_subset:
                print("✓ ERA5 grid points are subset of geopotential grid (using nearest-neighbor selection)")
                # Use sel with nearest method - this is like a left join on coordinates
                geo_ds = geo_ds.sel(latitude=lat_coords, longitude=lon_coords, method='nearest')
                print(f"  Selected geopotential dimensions: {geo_ds.dims}")
            else:
                print("⚠ WARNING: Geopotential grid does not contain all ERA5 points!")
                print("  Falling back to linear interpolation...")
                geo_ds = geo_ds.interp(latitude=lat_coords, longitude=lon_coords, method='linear')
                print(f"  Interpolated geopotential dimensions: {geo_ds.dims}")

        # Convert geopotential to altitude: altitude = geopotential / g
        # Geopotential is in m²/s², g = 9.80665 m/s²
        GRAVITY = 9.80665
        geopotential = geo_ds[geopotential_var_name]

        # If there's a time dimension, take first timestep (geopotential is static)
        if 'time' in geopotential.dims:
            geopotential = geopotential.isel(time=0)
            print("  Selected first time step from geopotential (static field)")

        altitude = geopotential / GRAVITY
        print(f"Converted geopotential to altitude (m). Range: [{float(altitude.min()):.1f}, {float(altitude.max()):.1f}] m")

        # Convert to torch tensor
        # altitude_tensor = torch.from_numpy(altitude.values.astype(np.float32)).to(device)
        # print(f"Altitude tensor shape: {altitude_tensor.shape}, dtype: {altitude_tensor.dtype}")

    return tensor_Z, stats, altitude, time_coords


# Load high-resolution topography data
def load_high_res_topography(
    topography_path: Path | str,
    dem_var: str = 'DEM',
    tpi_var: str = 'TPI_500M',
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Loads high-resolution topography data (DEM and TPI) from a Zarr file.

    Note: the dataset we used mixed Zarr format 3 and Zarr format 2 and that
    caused trouble. Renaming zarr.json to zarr.json.bak makes the dataset
    fully loadable in format 2.

    Args:
        topography_path: Path to the Zarr directory containing topography data
        dem_var: Name of the DEM variable (default 'DEM')
        tpi_var: Name of the TPI variable (default 'TPI_500M')

    Returns:
        Tuple of (dem_data, tpi_data) as xarray.DataArrays
    """
    print(f"Loading high-resolution topography from Zarr: {topography_path}")
    topo_ds = xr.open_zarr(str(topography_path))
    print(f"Topography dataset dimensions: {topo_ds.dims}")

    # Load DEM
    if dem_var in topo_ds:
        dem_data = topo_ds[dem_var]
    else:
        first_var = list(topo_ds.data_vars)[0]
        print(f"Warning: '{dem_var}' variable not found. Using first variable '{first_var}' instead.")
        dem_data = topo_ds[first_var]
    print(f"DEM shape: {dem_data.shape}, dtype: {dem_data.dtype}")

    # Load TPI
    if tpi_var in topo_ds:
        tpi_data = topo_ds[tpi_var]
        print(f"TPI ({tpi_var}) shape: {tpi_data.shape}, dtype: {tpi_data.dtype}")
    else:
        print(f"Warning: '{tpi_var}' variable not found. TPI will be None.")
        tpi_data = None

    return dem_data, tpi_data


def prepare_meteoswiss_targets(
    meteo_swiss_glob: str,
    normalization_stats: Era5Metadata,
    data_var: str = 'TmaxD',
    grid_elevation: xr.DataArray | None = None,
    hi_res_elevation: xr.DataArray | None = None,
    hi_res_tpi: xr.DataArray | None = None,
    convert_to_kelvin: bool = False,
    year_start: int | None = None,
    device: torch.device | None = None,
) -> Tuple[xr.DataArray, xr.DataArray, torch.Tensor]:
    """
    Transforms the MeteoSwiss Dataset into the Target Tensors (x, y, e).

    Args:
        meteo_swiss_glob: Glob pattern for MeteoSwiss NetCDF files
        normalization_stats: Stats from input for normalization
        data_var: Name of temperature variable (default 'TmaxD')
        grid_elevation: xarray.DataArray with ERA5 grid elevation (optional)
        hi_res_elevation: xarray.DataArray with high-res topography (optional)
        hi_res_tpi: xarray.DataArray with high-res TPI data (optional, e.g. TPI_2000M)
        convert_to_kelvin: Set True if MeteoSwiss is Celsius and input is Kelvin
        year_start: Optional year to filter data from
        device: Torch device to load elevation tensor to (defaults to CPU)

    Returns:
        X (Locations): xr.DataArray (point, 2) -> [Lat_norm, Lon_norm]
        Y (Truth):     xr.DataArray (time, point) -> [Temp_norm]
        E (Topo):      torch.Tensor (point, 3) -> [True_Elev, Elev_Diff, mTPI]
    """
    ds = xr.open_mfdataset(str(meteo_swiss_glob), combine='by_coords', data_vars='all')
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
    lat_range = normalization_stats.lat_max - normalization_stats.lat_min
    lon_range = normalization_stats.lon_max - normalization_stats.lon_min
    if lat_range == 0 or lon_range == 0:
        raise ValueError(f"Normalization coordinate range is zero (lat_range={lat_range}, lon_range={lon_range}); cannot normalize.")
    lat_norm = (lat_flat - normalization_stats.lat_min) / lat_range
    lon_norm = (lon_flat - normalization_stats.lon_min) / lon_range

    # Stack into (point, 2)
    tensor_x = xr.concat([lat_norm, lon_norm], dim="coord")
    tensor_x = tensor_x.transpose("point", "coord")
    tensor_x = tensor_x.assign_coords(coord=["lat", "lon"])

    # 3. Prepare Ground Truth Data (Tensor y)
    print(f"Creating Ground Truth Tensor (y) for {data_var}...")

    data_flat = ds_flat[data_var]

    # --- UNIT CONVERSION CHECK ---
    if convert_to_kelvin:
        print(f"  -> Converting Celsius to Kelvin (+{KELVIN_OFFSET}) before normalization")
        data_flat = data_flat + KELVIN_OFFSET

    # Normalize using stats (Z-Score)
    # Truth = (Temp - Input_Mean) / Input_Std
    if normalization_stats.data_std == 0:
        raise ValueError("Normalization data_std is zero (constant field); cannot normalize targets.")
    y_norm = (data_flat - normalization_stats.data_mean) / normalization_stats.data_std

    tensor_y = y_norm.transpose("time", "point")

    # 4. Prepare Topography (Tensor e)
    # The paper expects 3 features: [True Elevation, Elevation Diff, mTPI]
    print("Creating Elevation Tensor (e) [3 features]...")

    feature_names = ['true_elev', 'elev_diff', 'mTPI']

    if hi_res_elevation is not None and grid_elevation is not None:
        # Get target coordinates as numpy arrays
        lat_vals = lat_flat.values if hasattr(lat_flat, 'values') else lat_flat.compute().values
        lon_vals = lon_flat.values if hasattr(lon_flat, 'values') else lon_flat.compute().values

        # Convert WGS84 lat/lon to Swiss LV95 (x, y) for hi-res DEM lookup
        print("  -> Converting target coordinates to Swiss LV95 for hi-res DEM lookup...")
        target_x_lv95, target_y_lv95 = wgs84_to_lv95(lon_vals, lat_vals)

        # Interpolate hi-res DEM at target points using LV95 coordinates
        # The hi-res DEM has dimensions (y, x) with x, y as the index coordinates
        print("  -> Interpolating hi-res DEM at target points...")
        true_elev = hi_res_elevation.interp(
            x=xr.DataArray(target_x_lv95, dims='point'),
            y=xr.DataArray(target_y_lv95, dims='point'),
            method='linear'
        )

        # Interpolate ERA5 grid elevation at target points using lat/lon
        # ERA5 grid elevation has dimensions (latitude, longitude)
        print("  -> Interpolating ERA5 grid elevation at target points...")
        grid_elev = grid_elevation.interp(
            latitude=xr.DataArray(lat_vals, dims='point'),
            longitude=xr.DataArray(lon_vals, dims='point'),
            method='linear'
        )

        # Compute elevation difference
        elev_diff = true_elev - grid_elev

        print(f"  -> True elevation range: [{float(true_elev.min()):.1f}, {float(true_elev.max()):.1f}] m")
        print(f"  -> Grid elevation range: [{float(grid_elev.min()):.1f}, {float(grid_elev.max()):.1f}] m")
        print(f"  -> Elevation diff range: [{float(elev_diff.min()):.1f}, {float(elev_diff.max()):.1f}] m")

        # Interpolate TPI at target points if provided
        if hi_res_tpi is not None:
            print("  -> Interpolating TPI at target points...")
            tpi = hi_res_tpi.interp(
                x=xr.DataArray(target_x_lv95, dims='point'),
                y=xr.DataArray(target_y_lv95, dims='point'),
                method='linear'
            )
            print(f"  -> TPI range: [{float(tpi.min()):.1f}, {float(tpi.max()):.1f}] m")
        else:
            print("  -> TPI data not provided. Filling with zeros.")
            tpi = xr.zeros_like(true_elev)

        # Stack: [True Elev, Elev Diff, mTPI]
        tensor_e = xr.concat([true_elev, elev_diff, tpi], dim="feature", coords='minimal')
    else:
        print("  -> Warning: Elevation data not provided. Filling ALL 3 features with zeros.")
        zeros = xr.zeros_like(lat_flat)
        tensor_e = xr.concat([zeros, zeros, zeros], dim="feature")

    # Assign coordinates and transpose to (point, feature)
    tensor_e = tensor_e.assign_coords(feature=feature_names)
    tensor_e = tensor_e.transpose("point", "feature")

    # Convert elevation to torch tensor
    if device is None:
        device = torch.device('cpu')
    tensor_e_torch = torch.from_numpy(tensor_e.values.astype(np.float32)).to(device)

    print(f"Final X (Locations): {tensor_x.sizes}")
    print(f"Final Y (Targets):   {tensor_y.sizes}")
    print(f"Final E (Topo):      {tensor_e_torch.shape}, dtype: {tensor_e_torch.dtype}")

    return tensor_x, tensor_y, tensor_e_torch


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


def compute_seasonal_features(
    time_coords: np.ndarray,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute seasonal features (cos/sin of day-of-year) from time coordinates.

    The seasonal encoding allows the elevation MLP to learn season-dependent
    corrections, e.g., different lapse rates in winter (inversions) vs summer.

    Args:
        time_coords: Array of datetime64 values (from load_era5_data)
        device: Torch device to load tensor to (defaults to CPU)

    Returns:
        Tensor of shape (n_times, 2) with [cos(day_of_year), sin(day_of_year)]
    """
    if device is None:
        device = torch.device('cpu')

    # Convert datetime64 to day of year
    times_pd = pd.to_datetime(time_coords)
    day_of_year = times_pd.dayofyear.values  # 1-366

    # Convert to radians (0 to 2*pi over the year)
    time_rads = (day_of_year - 1) / 365.0 * 2 * np.pi

    # Compute cos and sin encoding
    cos_doy = np.cos(time_rads)
    sin_doy = np.sin(time_rads)

    # Stack into (n_times, 2) tensor
    seasonal = np.stack([cos_doy, sin_doy], axis=1)
    seasonal_tensor = torch.from_numpy(seasonal.astype(np.float32)).to(device)

    print(f"Computed seasonal features: {seasonal_tensor.shape}, dtype: {seasonal_tensor.dtype}")

    return seasonal_tensor


def interpolate_era5_to_targets(
    y_context: torch.Tensor,
    target_coords: xr.DataArray,
    metadata: Era5Metadata,
) -> torch.Tensor:
    """
    Bilinearly interpolate ERA5 temperature (channel 0) to target point locations.

    This provides the reference (baseline) predictions for skill score calculation:
    the best you could do by simply reading off the coarse ERA5 grid at each station.

    Args:
        y_context: ERA5 input tensor of shape (time, channels, lat, lon).
                   Channel 0 is the normalized temperature field.
        target_coords: xarray.DataArray of shape (point, coord) with normalized
                       [lat, lon] coordinates in [0, 1] (from prepare_meteoswiss_targets).
        metadata: Era5Metadata with lat_coords to determine grid orientation.

    Returns:
        Tensor of shape (time, n_points) with interpolated normalized temperatures.
    """
    n_time = y_context.shape[0]
    temp_field = y_context[:, 0:1, :, :]  # (time, 1, lat, lon)

    # Extract normalized target coordinates
    lat_norm = torch.from_numpy(
        target_coords.sel(coord='lat').values.astype(np.float32)
    )
    lon_norm = torch.from_numpy(
        target_coords.sel(coord='lon').values.astype(np.float32)
    )

    # Convert [0, 1] normalized coords to grid_sample's [-1, 1] range.
    # grid_sample convention: -1 maps to index 0, +1 maps to last index.
    # Longitude (W dimension): lon increases with index, so grid_x = 2*lon_norm - 1
    grid_x = 2.0 * lon_norm - 1.0

    # Latitude (H dimension): check ordering of the ERA5 grid.
    # If lat_coords is descending (north-to-south), index 0 = lat_max (norm=1),
    # so grid_y = -(2*lat_norm - 1) = 1 - 2*lat_norm.
    # If ascending (south-to-north), index 0 = lat_min (norm=0),
    # so grid_y = 2*lat_norm - 1.
    lat_descending = metadata.lat_coords[0] > metadata.lat_coords[-1]
    if lat_descending:
        grid_y = 1.0 - 2.0 * lat_norm
    else:
        grid_y = 2.0 * lat_norm - 1.0

    # grid_sample expects grid of shape (N, H_out, W_out, 2) with (x, y) ordering.
    # For point sampling: H_out=1, W_out=n_points
    n_points = len(lat_norm)
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (n_points, 2)
    grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, n_points, 2)
    grid = grid.expand(n_time, -1, -1, -1)  # (time, 1, n_points, 2)

    # Move to same device as input
    grid = grid.to(temp_field.device)

    # Bilinear interpolation
    interpolated = F.grid_sample(
        temp_field, grid, mode='bilinear', padding_mode='border', align_corners=True
    )
    # interpolated shape: (time, 1, 1, n_points) -> squeeze to (time, n_points)
    return interpolated.squeeze(1).squeeze(1)
