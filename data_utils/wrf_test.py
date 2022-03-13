import os
import xarray as xr
import numpy as np
import scipy.interpolate.ndgriddata as ndgriddata
import netCDF4 as nc
from datetime import date, timedelta
from plot import *

"""
Testing preprocessing for the wrf datasets
"""

REGION_COORDS = {'nwus':([38,48],[238,248]), 'seus':([28,38], [268,278]), 
				 'neus':([35,45], [273,283]), 'swus':([32,42], [242,252]),
				 'mwus':([35,45],[255, 265])}

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

def wrf_time_downsample(infile='./ncdata/wrf-200010-200012-precip-raw.nc', var_name='PREC_ACC_NC', 
				   		time='24H', method='sum', outfile= './ncdata/wrf-200010-200012-precip.nc'):
	wrf_data = xr.open_dataset(infile)
	print(wrf_data)
	wrf_var = wrf_data[var_name]
	if method=='sum':
		wrf_var = wrf_var.resample(Time=time).sum('Time')
	elif method=='max':
		wrf_var = wrf_var.resample(Time=time).max()
	else:
		assert(False)
	wrf_var.to_netcdf(outfile)
	print(f'{infile} downsampled to {time}')

# # initial downsampling to daily sum
# var_name = 'temp'
# date_range='200610-200612'
# wrf_time_downsample(infile=f'../ncdata/wrf-{date_range}-{var_name}-raw.nc', outfile=f'../ncdata/wrf-{date_range}-{var_name}.nc',
# 					method='max', var_name='T2')

# loading in daily max temp
wrf_data = xr.open_dataset('../ncdata/wrf-200010-200012-temp.nc')
print(wrf_data)
wrf_var = wrf_data['T2']
print(wrf_var)

# lat_range, lon_range = REGION_COORDS['nwus']
# interp_wrf_var = wrf_interpolate(wrf_var, lat_range, lon_range, steps=100)
# print(interp_wrf_var)
# plot_image_map(interp_wrf_var[0,:,:], interp_wrf_var.lat, interp_wrf_var.lon)
# plt.show()

# # coordinates are not exactly aligned to a grid
# print(wrf_temp.XLAT[:,0]) 
# print(wrf_temp.XLAT[0,:])
# plot_image_map(wrf_temp[0,:,:], wrf_temp.XLAT[:,0], wrf_temp.XLONG[0,:])
# plt.show()

# # Plot actual coordinates
# wrf_lats = wrf_temp.XLAT.values.flatten() 
# wrf_lons = wrf_temp.XLONG.values.flatten() + 360
# plt.scatter(wrf_lons, wrf_lats)
# plt.show()