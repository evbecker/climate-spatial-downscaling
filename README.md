# Downscaling Project

### Setup
Necessary python packages can be found in environment.yaml and installed with conda directly 

Netcdf files currently located in private directory, consist of ERA-iterim, NOAA CPC, and high resolution WRF datasets. 
Please email evbecker @ ucla.edu for access or download manually from:
-[WRF](https://rda.ucar.edu/datasets/ds612.0)
-[ERA-Iterim](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era-interim)
-[CPC](https://psl.noaa.gov/data/gridded/data.unified.daily.conus.html)

### Overview
The module preprocessing.py contains most of the functions necessary for converting netcdf datasets to torch tensors
