import os
import xarray as xr
import numpy as np
import netCDF4 as nc
import torch
import matplotlib.pyplot as plt
from plot import *

"""
Testing interpolation methods for CPC and ERA-iterim datasets
"""

REGION_COORDS = {'nwus':([38,48],[238,248]), 'seus':([28,38], [268,278]), 
				 'neus':([35,45], [273,283]), 'swus':([32,42], [242,252])}

def crop_to_region(data, lat_range, lon_range):
	min_lat, max_lat = lat_range
	min_lon, max_lon = lon_range
	mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
	mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
	return data.where(mask_lat & mask_lon, drop =True)

def interpolate(data, lat_range, lon_range, steps, method):
	lats = np.linspace(lat_range[0], lat_range[1], steps)
	lons = np.linspace(lon_range[0], lon_range[1], steps)
	interp_data = data.interp(lat=lats, lon=lons, method=method)
	interp_data = interp_data.fillna(0)
	return interp_data

if __name__ == "__main__":
	# loading in data
	data_hr = xr.open_dataset('./ncdata/cpc-2000-precip.nc')
	data_hr = data_hr.fillna(0)
	precip_hr = data_hr['precip']
	data_lr = xr.open_dataset('./ncdata/erai-2000-precip.nc')
	data_lr = data_lr.fillna(0)
	data_lr = data_lr.rename({'longitude':'lon', 'latitude':'lat'})
	precip_lr = data_lr.tp.resample(time='24H').sum('time')*1000

	# create subplots
	fig, axarr, plot_next = image_map_factory(2,2, hspace=0.1);

	lat_range, lon_range = REGION_COORDS['swus']
	steps=40

	nwin_precip_lr = crop_to_region(precip_lr, lat_range,lon_range)
	plot_next(axarr[0,0], nwin_precip_lr[10,:,:], nwin_precip_lr.lat, nwin_precip_lr.lon, min_max=[0,60], title='Low Resolution Daily Precipitation')

	nwin_precip_hr = crop_to_region(precip_hr,lat_range, lon_range)
	plot_next(axarr[0,1],nwin_precip_hr[10,:,:],nwin_precip_hr.lat, nwin_precip_hr.lon, min_max=[0,60], title='High Resolution Daily Precipitation')


	lr_upscale = interpolate(precip_lr, lat_range, lon_range, steps=steps, method='linear')
	plot_next(axarr[1,0], lr_upscale[10,:,:], lr_upscale.lat, lr_upscale.lon, min_max=[0,60], title='LR Interpolated Daily Precipitation')

	hr_upscale = interpolate(precip_hr, lat_range, lon_range, steps=steps, method='linear')
	plot_next(axarr[1,1], hr_upscale[10,:,:], hr_upscale.lat, hr_upscale.lon, min_max=[0,60], title='HR Interpolated Daily Precipitation')
	plt.show()





