import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from plot import *

REGION_COORDS = {'nwus':([38,48],[238,248]), 'seus':([28,38], [268,278]), 
				 'neus':([35,45], [273,283]), 'swus':([32,42], [242,252]),
				 'mwus':([35,45],[255, 265])}

data =  EraiCpcWrfDataset('../tensordata-precip-160', regions=['neus'])

# Plotting three images to compare
img_choice = 0
steps = 160
fig, axarr, plot_next = image_map_factory(1,3, hspace=0.1,cbar_per_subplot=True, 
										  gridlines=False, cbar_orientation='vertical')
lat_range, lon_range = REGION_COORDS['nwus']
lats = np.linspace(lat_range[0], lat_range[1], steps)
lons = np.linspace(lon_range[0], lon_range[1], steps)
curr_precip_data = data.__getitem__(img_choice)
erai_img = curr_precip_data['slr_img']
cpc_img = curr_precip_data['lr_img']
wrf_img = curr_precip_data['hr_img']
max_value = torch.max(torch.cat((erai_img,cpc_img,wrf_img)))

plot_next(axarr[0], erai_img.numpy(), lats, lons, title='ERAI Daily Precipitation',
			min_max=[0, max_value])
plot_next(axarr[1], cpc_img.numpy(), lats, lons, title='CPC Resolution Daily Precipitation',
			min_max=[0, max_value])
plot_next(axarr[2], wrf_img.numpy(), lats, lons, title='WRF Resolution Daily Precipitation',
			min_max=[0, max_value])

print(curr_precip_data['name'])
plt.suptitle('Precipitation (MM) on 01/01/2003 in NE US', fontsize=18)
plt.show()