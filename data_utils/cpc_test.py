import os
import xarray as xr
import numpy as np
import netCDF4 as nc
import torch
import pandas as pd
from datetime import date, timedelta
import scipy.interpolate.ndgriddata as ndgriddata

"""
Testing preprocessing for the two cpc datasets
"""

MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul','aug', 'sep', 'oct', 'nov', 'dec']
YEARS = ['2000', '2001', '2002', '2003', '2004', '2005', '2006']
QUARTERS = [('01','03'), ('04','06'), ('07','09'), ('10','12')]
REGION_COORDS = {'nwus':([38,48],[238,248]), 'seus':([28,38], [268,278]), 
				 'neus':([35,45], [273,283]), 'swus':([32,42], [242,252]),
				 'mwus':([35,45],[255, 265])}

file="./ncdata/cpc-2007-precip.nc"
out_dir= './tensordata-precip-40'
region='nwus'
steps=40

# assuming netcdf file has latitude, longitude, time, and var_name
# converts image at each timepoint to a tensor
cpc_data = xr.open_dataset(file)
cpc_data = cpc_data.fillna(0)
time = cpc_data.time.dt.date
print(time.values[0])
# if file is from after 2006, we need to downsample to daily measurements
if time.values[0]>date(2006, 12, 31):
	cpc_precip = cpc_data.precip.resample(time='24H').sum('time')
else:
	cpc_precip = cpc_data.precip
print(cpc_precip)
# cropping data to specific region
lat_range, lon_range = REGION_COORDS[region]
if steps == None:
	regional_precip = crop_to_region(cpc_precip, lat_range, lon_range)
else:
	regional_precip = interpolate(cpc_precip, lat_range, lon_range, steps, method='nearest')
# saving tensor "image" for each time point
for i, t in enumerate(time.values):
	print(f'time: {t}, with shape: {regional_precip[i,:,:].shape}')
	out_path = os.path.join(out_dir, f'cpc-{region}-precip-{t}.pt')
	torch.save(torch.tensor(regional_precip[i,:,:].values), out_path) 