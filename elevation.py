import xarray as xr
import numpy as np
import torch
import netCDF4 as nc
from datetime import date, timedelta
from plot import *

"""
Getting approximate elevation data for different US regions
Takes in a netcdf file and saves output as torch tensor
"""

REGION_COORDS = {'nwus':([38,48],[238,248]), 'seus':([28,38], [268,278]), 
				 'neus':([35,45], [273,283]), 'swus':([32,42], [242,252]),
				 'mwus':([35,45],[255, 265])}

def interpolate(data, lat_range, lon_range, steps, method):
	lats = np.linspace(lat_range[0], lat_range[1], steps)
	lons = np.linspace(lon_range[0], lon_range[1], steps)
	interp_data = data.interp(lat=lats, lon=lons, method=method)
	interp_data = interp_data.fillna(0)
	return interp_data

# loading in elevation data
elev_data = xr.open_dataset('./ncdata/elev-americas-5-min.nc')
elevations = elev_data.data[0,:,:]
plot_image_map(elevations, elevations.lat, elevations.lon, cmap='inferno')
plt.show()

# interpolating to desired coordinates
regions = ['nwus', 'swus', 'mwus', 'neus', 'seus']
for region in regions:
	lat_range, lon_range = REGION_COORDS[region]
	steps=160

	interp_elev = interpolate(elevations, lat_range, lon_range, steps, method='linear')
	# plot_image_map(interp_elev, interp_elev.lat, interp_elev.lon, cmap='inferno')
	# plt.show()

	# save this as a tensor
	torch.save(torch.tensor(interp_elev.values), f'./tensordata-precip-{steps}/{region}-elevation-{steps}x{steps}.pt')