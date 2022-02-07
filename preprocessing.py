import os
import xarray as xr
import numpy as np
import netCDF4 as nc
import torch
import pandas as pd
from datetime import date, timedelta

"""
TODO: looks like missing data might have to be handled
TODO: erai data on first and last days of month only have 12 hr totals
TODO: need to interpolate images to be on the same coords?

"""

REGION_COORDS = {'nwus':([34,48],[238,248])}

# returns only values corresponding to coords within given limits
def crop_to_region(data, lat_range, lon_range):
	min_lat, max_lat = lat_range
	min_lon, max_lon = lon_range
	mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
	mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
	return data.where(mask_lat & mask_lon, drop =True)

def erai_precip_to_torch_tensors(file='./erai-jan2000-precip.nc', out_dir= './data', region='nwus'):
	# assuming netcdf file has latitude, longitude, time, and var_name
	# converts image at each timepoint to a tensor
	erai_data = xr.open_dataset(file)
	erai_data = erai_data.rename({'longitude':'lon', 'latitude':'lat'})
	erai_data = erai_data.fillna(0)
	# summing 12hr periods to 24 hr totals, converting to mm
	erai_precip = erai_data.tp.resample(time='24H').sum('time')*1000
	time = erai_precip.time.dt.date
	# cropping data to specific region
	lat_range, lon_range = REGION_COORDS[region]
	regional_precip = crop_to_region(erai_precip, lat_range, lon_range)
	# saving tensor "image" for each time point
	for i, t in enumerate(time.values):
		out_path = os.path.join(out_dir, f'erai-{region}-precip-{t}.pt')
		print(f'time: {t}, with shape: {regional_precip[i,:,:].shape}')
		torch.save(torch.tensor(regional_precip[i,:,:].values), out_path) 

def cpc_precip_to_torch_tensors(file='./cpc-2000-precip.nc', out_dir= './data', region='nwus'):
	# assuming netcdf file has latitude, longitude, time, and var_name
	# converts image at each timepoint to a tensor
	cpc_data = xr.open_dataset(file)
	cpc_data = cpc_data.fillna(0)
	cpc_precip = cpc_data.precip
	time = cpc_data.time.dt.date
	# cropping data to specific region
	lat_range, lon_range = REGION_COORDS[region]
	regional_precip = crop_to_region(cpc_precip, lat_range, lon_range)
	# saving tensor "image" for each time point
	for i, t in enumerate(time.values):
		print(f'time: {t}, with shape: {regional_precip[i,:,:].shape}')
		out_path = os.path.join(out_dir, f'cpc-{region}-precip-{t}.pt')
		torch.save(torch.tensor(regional_precip[i,:,:].values), out_path) 

def make_precip_csv(sdate, edate, region='nwus', out_dir='./data'):
	# saves csv of metadata for combined erai, cpc dataset
	datetimes = pd.date_range(sdate,edate-timedelta(days=1),freq='d')
	dates = datetimes.date
	erai = np.array([f'erai-{region}-precip-{date}.pt' for date in dates])
	cpc = np.array([f'cpc-{region}-precip-{date}.pt' for date in dates])
	df = pd.DataFrame([dates, erai, cpc])
	df.to_csv(os.path.join(out_dir, f'{region}-{sdate}.csv'), header=False, index=False)


if __name__ == "__main__":
	# erai_precip_to_torch_tensors(file='./ncdata/erai-jan2000-precip.nc', out_dir='./tensordata')
	cpc_precip_to_torch_tensors(file = './ncdata/cpc-2000-precip.nc', out_dir='./tensordata')
	# make_precip_csv(sdate=date(2000,1,1), edate=date(2000,12,31), out_dir='./tensordata')