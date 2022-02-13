import os
import xarray as xr
import numpy as np
import netCDF4 as nc
import torch
import pandas as pd
from datetime import date, timedelta

"""
TODO: erai data on first and last days of month only have 12 hr totals

"""
MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul','aug', 'sep', 'oct', 'nov', 'dec']
REGION_COORDS = {'nwus':([38,48],[238,248])}

# returns only values corresponding to coords within given limits
def crop_to_region(data, lat_range, lon_range):
	min_lat, max_lat = lat_range
	min_lon, max_lon = lon_range
	mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
	mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
	return data.where(mask_lat & mask_lon, drop =True)

# interpolates values to new coordinates using scipy method
def interpolate(data, lat_range, lon_range, steps, method="nearest"):
	lats = np.linspace(lat_range[0], lat_range[1], steps)
	lons = np.linspace(lon_range[0], lon_range[1], steps)
	interp_data = data.interp(lat=lats, lon=lons, method=method)
	interp_data = interp_data.fillna(0)
	return interp_data

def erai_precip_to_torch_tensors(file='./erai-jan2000-precip.nc', out_dir='./tensordata', 
								 region='nwus', steps=40):
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
	if steps == None:
		regional_precip = crop_to_region(erai_precip, lat_range, lon_range)
	else:
		regional_precip = interpolate(erai_precip, lat_range, lon_range, steps=steps, method='nearest')
	# saving tensor "image" for each time point
	for i, t in enumerate(time.values):
		out_path = os.path.join(out_dir, f'erai-{region}-precip-{t}.pt')
		print(f'time: {t}, with shape: {regional_precip[i,:,:].shape}')
		torch.save(torch.tensor(regional_precip[i,:,:].values), out_path) 

def cpc_precip_to_torch_tensors(file='./cpc-2000-precip.nc', out_dir= './tensordata', 
								region='nwus', steps=40):
	# assuming netcdf file has latitude, longitude, time, and var_name
	# converts image at each timepoint to a tensor
	cpc_data = xr.open_dataset(file)
	cpc_data = cpc_data.fillna(0)
	cpc_precip = cpc_data.precip
	time = cpc_data.time.dt.date
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

def make_precip_csv(sdate, edate, region='nwus', out_dir='./data'):
	# saves csv of metadata for combined erai, cpc dataset
	# assumes we have daily data from start date to end date
	datetimes = pd.date_range(sdate,edate-timedelta(days=1),freq='d')
	dates = datetimes.date
	erai = np.array([f'erai-{region}-precip-{date}.pt' for date in dates])
	cpc = np.array([f'cpc-{region}-precip-{date}.pt' for date in dates])
	df = pd.DataFrame([dates, erai, cpc])
	df.to_csv(os.path.join(out_dir, f'{region}-{sdate}.csv'), header=False, index=False)


if __name__ == "__main__":

	for month in MONTHS:
		erai_precip_to_torch_tensors(file=f'./ncdata/erai-{month}2000-precip.nc', out_dir='./tensordata')
	cpc_precip_to_torch_tensors(file = './ncdata/cpc-2000-precip.nc', out_dir='./tensordata')
	# make_precip_csv(sdate=date(2000,1,1), edate=date(2000,12,31), out_dir='./tensordata')