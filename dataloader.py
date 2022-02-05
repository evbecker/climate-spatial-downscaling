import os
import netCDF4 as nc
import torch
import matplotlib.pyplot as plt
from plot import plot_image_map

"""
TODO: looks like missing data, scaling needs to be handled first
TODO: need to upscale images to match NF paper (seems like torch 
	  may have a built-in method for this)

"""

def test_erai_data(file='./jan18-temps.nc'):
	data = nc.Dataset(file)
	lat = data['latitude'][:]
	lon = data['longitude'][:]
	plot_image_map(temp[0,:,:], lat, lon)
	plt.show()

def erai_to_torch_tensors(file='./jan18-temps.nc', out_dir= './data', var_name='mx2t', region=''):
	# assuming netcdf file has latitude, longitude, time, and var_name
	# converts image at each timepoint to a tensor
	data = nc.Dataset(file)
	lat = data['latitude'][:]
	lon = data['longitude'][:]
	time = data['time'][:]
	var = data[var_name][:]
	for i, t in enumerate(time):
		out_path = os.path.join(out_dir, f'erai{region}-{var_name}-{t}.pt')
		torch.save(torch.tensor(var[i,:,:]), out_path) 
