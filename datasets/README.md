# Datasets used

## Sources

* EOBS: 
  https://www.ecad.eu/download/ensembles/download.php

* ERA5-Land: 
  https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation

* MeteoSwiss grid-data products:
  https://www.meteoswiss.admin.ch/dam/jcr:818a4d17-cb0c-4e8b-92c6-1a1bdf5348b7/ProdDoc_TabsD.pdf
  https://www.meteoswiss.admin.ch/dam/jcr:4f51f0f1-0fe3-48b5-9de0-15666327e63c/ProdDoc_RhiresD.pdf


## Local storage

Download and store them in the following folder structure:

- datasets
  - EOBS [not currently used in project]
    - max_temperature
      - t_max_day_EOBS_v30_CH_1971-2024.nc
    - min_temperature
      - t_min_day_EOBS_v30_CH_1971-2024.nc
    - precipitation
      - pr_day_EOBS_v30_CH_1971-2024.nc
    - temperature
      - tas_day_EOBS_v30_CH_1971-2024.nc
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


## Dependencies

You will need `xarray` and `h5netcdf`.

```
pip install xarray h5netcdf
```


## Loading

Load the data like this:

```
import xarray as xr

eobs_temp2m = xr.open_dataset("datasets/EOBS/temperature/tas_day_EOBS_v30_CH_1971-2024.nc")
era5_temp2m = xr.open_dataset("datasets/ERA5_Land/temperature/t2m-1960.nc")
mch_temp2m = xr.open_dataset("datasets/MeteoSwiss/TabsD_v2.0_swiss.lv95/TabsD_ch01r.swiss.lv95_196101010000_196112310000.nc")
```
