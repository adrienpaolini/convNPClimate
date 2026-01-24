# Datasets used

## Sources

* ERA5-Land: 
  https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation

* MeteoSwiss grid-data products:
  https://www.meteoswiss.admin.ch/dam/jcr:818a4d17-cb0c-4e8b-92c6-1a1bdf5348b7/ProdDoc_TabsD.pdf
  https://www.meteoswiss.admin.ch/dam/jcr:4f51f0f1-0fe3-48b5-9de0-15666327e63c/ProdDoc_RhiresD.pdf


## Local storage

Download and store them in the following folder structure:

- datasets
  - ERA5_Land
    - max_temperature
      - t2m_max-1960.nc
      - t2m_max-1961.nc
      - ...
      - t2m_max-2023.nc
    - min_temperature
      - t2m_min-1960.nc
      - t2m_min-1961.nc
      - ...
      - t2m_min-2023.nc
    - precipitation
      - tp-1960.nc
      - tp-1961.nc
      - ...
      - tp-2023.nc
    - temperature
      - t2m-1960.nc
      - t2m-1961.nc
      - ...
      - t2m-2023.nc
    - geopotential
      - era5_land_geopotential.nc
  - MeteoSwiss
    - RhiresD_v2.0_swiss.lv95
      - RhiresD_ch01h.swiss.lv95_196101010000_196112310000.nc
      - RhiresD_ch01h.swiss.lv95_196201010000_196212310000.nc
      - ...
      - RhiresD_ch01h.swiss.lv95_202301010000_202312310000.nc
    - TabsD_v2.0_swiss.lv95
      - TabsD_ch01r.swiss.lv95_196101010000_196112310000.nc
      - TabsD_ch01r.swiss.lv95_196201010000_196212310000.nc
      - ...
      - TabsD_ch01r.swiss.lv95_202301010000_202312310000.nc
    - TmaxD_v2.0_swiss.lv95
      - TmaxD_ch01r.swiss.lv95_197101010000_197112310000.nc
      - TmaxD_ch01r.swiss.lv95_197201010000_197212310000.nc
      - ...
      - TmaxD_ch01r.swiss.lv95_202301010000_202312310000.nc
    - TminD_v2.0_swiss.lv95
      - TminD_ch01r.swiss.lv95_197101010000_197112310000.nc
      - TminD_ch01r.swiss.lv95_197201010000_197212310000.nc
      - ...
      - TminD_ch01r.swiss.lv95_202301010000_202312310000.nc
  - topo_subset.zarr (all subfolders with content in zarray format 2)
    - DEM
    - EASTING
    - latitude
    - longitude
    - NORTHING
    - SN_DERIVATIVE_2000M_SIGRATIO1
    - SN_DERIVATIVE_500M_SIGRATIO1
    - TPI_2000M
    - TPI_500M
    - VALLEY_NORM_2000M_SMTHFACT0.5
    - WE_DERIVATIVE_2000M_SIGRATIO1
    - WE_DERIVATIVE_500M_SIGRATIO1
    - x
    - y
    - zarr.json
    (Note: this dataset mixes Zarr format 3 and Zarr format 2 and that causes trouble. Renaming zarr.json to zarr.json.bak makes the dataset fully loadable in format 2.)


## Dependencies

You will need `xarray` and `h5netcdf`.

```
pip install xarray h5netcdf
```


## Loading

Directly load the data like this:

```
import xarray as xr

era5_temp2m = xr.open_dataset("datasets/ERA5_Land/temperature/t2m-1960.nc")
mch_temp2m = xr.open_dataset("datasets/MeteoSwiss/TabsD_v2.0_swiss.lv95/TabsD_ch01r.swiss.lv95_196101010000_196112310000.nc")
topo_ds = xr.open_zarr("datasets/topo_subset.zarr")
```

datasets.py contains tooling for loading and processing the datasets as well.