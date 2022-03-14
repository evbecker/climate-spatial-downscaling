import os
import argparse
import xarray as xr
import numpy as np
import netCDF4 as nc
import torch
import pandas as pd
from datetime import date, timedelta
import scipy.interpolate.ndgriddata as ndgriddata

"""
TODO: add argparser for final submission

"""
MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul','aug', 'sep', 'oct', 'nov', 'dec']
YEARS1 = ['2000', '2001', '2002', '2003', '2004', '2005', '2006']
YEARS2 = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', 
		  '2014', '2015', '2016', '2017', '2018']
QUARTERS = [('01','03'), ('04','06'), ('07','09'), ('10','12')]
REGION_COORDS = {'nwus':([38,48],[238,248]), 'seus':([28,38], [268,278]), 
				 'neus':([35,45], [273,283]), 'swus':([32,42], [242,252]),
				 'mwus':([35,45],[255, 265])}

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

# wrf data is not aligned in a regular grid so needs special interpolation method
def wrf_interpolate(data, lat_range, lon_range, steps, method="nearest"):
	# first retrieving original coords, flattening data
	data_lats = data.XLAT.values.flatten() 
	data_lons = data.XLONG.values.flatten() + 360
	data_flattened = np.reshape(data.values, (data.shape[0], np.prod(data.shape[1:])))
	# setting new coords to interpolate to
	lats = np.linspace(lat_range[0], lat_range[1], steps)
	lons = np.linspace(lon_range[0], lon_range[1], steps)
	X,Y = np.meshgrid(lons, lats)
	# using scipy method
	interp_array = np.zeros((data.shape[0], steps, steps))
	for i, t in enumerate(data.Time):
		interp_array[i,:,:] = ndgriddata.griddata((data_lons, data_lats), 
	                    					data_flattened[i], (X, Y), method=method)
	# converting back to xarray style array
	interp_data = xr.DataArray(data=interp_array,
							   dims=['time', 'lon', 'lat'],
							   coords=dict( lon=('lon', lons),
							   				lat=('lat', lats),
							   				time=('time',data.Time))
							   )
	return interp_data

# wrf files are large, so downsampling and saving to smaller file first
def wrf_time_downsample(infile='./ncdata/wrf-200010-200012-precip-raw.nc', var_name='PREC_ACC_NC', 
				   		time='24H', method='sum', outfile= './ncdata/wrf-200010-200012-precip.nc'):
	wrf_data = xr.open_dataset(infile)
	wrf_data = wrf_data.fillna(0)
	wrf_var = wrf_data[var_name]
	if method=='sum':
		wrf_var = wrf_var.resample(Time=time).sum('Time')
	elif method=='max':
		wrf_var = wrf_var.resample(Time=time).max()
	else:
		assert(False)
	wrf_var.to_netcdf(outfile)
	print(f'{infile} downsampled to {time}')

def wrf_to_torch_tensors(file='./wrf-200010-200012-precip.nc', out_dir='./tensordata', 
								 var='precip', region='nwus', steps=100):
	wrf_data = xr.open_dataset(file)
	if var == 'precip':
		wrf_var = wrf_data['PREC_ACC_NC']
	elif var == 'temp':
		wrf_var = wrf_data['T2']
	else:
		assert(False)
	time = wrf_var.Time.dt.date
	# cropping data to a specific region
	lat_range, lon_range = REGION_COORDS[region]
	regional_precip = wrf_interpolate(wrf_var, lat_range, lon_range, steps)
	for i, t in enumerate(time.values):
		out_path = os.path.join(out_dir, f'wrf-{region}-precip-{t}.pt')
		print(f'time: {t}, with shape: {regional_precip[i,:,:].shape}')
		torch.save(torch.tensor(regional_precip[i,:,:].values), out_path)


def erai_to_torch_tensors(file='./erai-jan2000-precip.nc', out_dir='./tensordata', 
								 var='precip', region='nwus', steps=40):
	# assuming netcdf file has latitude, longitude, time, and var_name
	# converts image at each timepoint to a tensor
	erai_data = xr.open_dataset(file)
	erai_data = erai_data.rename({'longitude':'lon', 'latitude':'lat'})
	erai_data = erai_data.fillna(0)
	if var=='precip':
		# summing 12hr periods to 24 hr totals, converting to mm
		erai_var = erai_data.tp.resample(time='24H').sum('time')*1000
	elif var=='temp':
		# taking the max of 12 hr samples to get 24 hr max
		erai_var = erai_data.mx2t.resample(time='24H').max()
	else:
		assert(False)
	time = erai_var.time.dt.date
	# cropping data to specific region
	lat_range, lon_range = REGION_COORDS[region]
	if steps == None:
		regional_var = crop_to_region(erai_var, lat_range, lon_range)
	else:
		regional_var = interpolate(erai_var, lat_range, lon_range, steps=steps, method='nearest')
	# saving tensor "image" for each time point
	for i, t in enumerate(time.values):
		out_path = os.path.join(out_dir, f'erai-{region}-{var}-{t}.pt')
		print(f'time: {t}, with shape: {regional_var[i,:,:].shape}')
		torch.save(torch.tensor(regional_var[i,:,:].values), out_path) 

def cpc_precip_to_torch_tensors(file='./cpc-2000-precip.nc', out_dir= './tensordata', 
								region='nwus', steps=40):
	# assuming netcdf file has latitude, longitude, time, and var_name
	# converts image at each timepoint to a tensor
	cpc_data = xr.open_dataset(file)
	cpc_data = cpc_data.fillna(0)
	time = cpc_data.time.dt.date
	# if file is from after 2006, we need to downsample to daily measurements
	if time.values[0]>date(2006, 12, 31):
		cpc_precip = cpc_data.precip.resample(time='24H').sum('time')
	else:
		cpc_precip = cpc_data.precip
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

def make_precip_csv(sdate, edate, erai_cpc=False, erai_wrf=False, cpc_wrf=False, 
					regions=['nwus'], out_dir='./data'):
	# saves csv of metadata for combined erai, cpc, wrf datasets
	# assumes we have daily data from start date to end date
	datetimes = pd.date_range(sdate,edate-timedelta(days=1),freq='d')
	dates = datetimes.date
	# creating file names for each dataset and region
	erai = np.array([f'erai-{region}-precip-{date}.pt' for date in dates for region in regions])
	cpc = np.array([f'cpc-{region}-precip-{date}.pt' for date in dates for region in regions])
	wrf = np.array([f'wrf-{region}-precip-{date}.pt' for date in dates for region in regions])
	if erai_cpc:
		df = pd.DataFrame([dates, erai, cpc])
	elif erai_wrf:
		df = pd.DataFrame([dates, erai, wrf])
	elif cpc_wrf:
		df = pd.DataFrame([dates, cpc, wrf])
	else:
		print('no datasets selected')
		assert(False)
	regions_id = '-'.join(regions)
	df.to_csv(os.path.join(out_dir, f'{regions_id}-{sdate}.csv'), header=False, index=False)


if __name__ == "__main__":
	var = 'temp'
	steps = 160
	regions = ['nwus', 'swus', 'mwus', 'neus', 'seus']

	# parse args
	parser = argparse.ArgumentParser()
	parser.add_argument('--var', type=str, required=True, help="variable type can be temp or precip")
	parser.add_argument('--steps', type=int, required=True, help="number of steps; max resolution is 10deg/steps")
	parser.add_argument('--datasets', type=list, required=False, default=['wrf','erai','cpc'])
	args = parser.parse_args()
	var = args.var
	steps = args.steps
	datasets = args.datasets

	if 'wrf' in datasets:
		for region in regions:
			for year in YEARS1[1:]:
				for quarter in QUARTERS:
					date_range=f'{year}{quarter[0]}-{year}{quarter[1]}'
					wrf_to_torch_tensors(file=f'../ncdata/wrf-{date_range}-{var}.nc', 
												out_dir=f'../tensordata-{var}-{steps}',
												var=var, region=region, steps=steps)			
	for region in regions:
		for year in YEARS1:
			if 'erai' in datasets:
				erai_to_torch_tensors(file=f'../ncdata/erai-{year}-{var}.nc', out_dir=f'../tensordata-{var}-{steps}', 
									  var=var, steps=steps, region=region)
			if 'cpc' in datasets:
				cpc_precip_to_torch_tensors(file = f'../ncdata/cpc-{year}-precip.nc', out_dir=f'./tensordata-{var}-{steps}', 
											steps=steps, region=region)