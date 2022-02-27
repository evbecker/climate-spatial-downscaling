import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from plot import *

"""
Comparing spatial averages of precipitation in the Northwest US
for the year 2000

"""
REGION_COORDS = {'nwus':([38,48],[238,248]), 'seus':([28,38], [268,278]), 
				 'neus':([35,45], [273,283]), 'swus':([32,42], [242,252])}

myData =  EraiCpcDataset('./tensordata/nwus-2000-01-01.csv', './tensordata')
lr_avg_precip = np.zeros(myData.__len__())
hr_avg_precip = np.zeros(myData.__len__())

for i in range(myData.__len__()):
	(hr_img, lr_img, prev_hr_img) = myData.__getitem__(i)
	lr_avg_precip[i] = torch.mean(lr_img).numpy()
	hr_avg_precip[i] = torch.mean(hr_img).numpy()


plt.plot(list(range(myData.__len__())), lr_avg_precip, label="low resolution")
plt.plot(list(range(myData.__len__())), hr_avg_precip, label="high resolution")
plt.xlabel("day of the year (2001)")
plt.ylabel("average precipitation (mm) for NW US")
plt.legend()
plt.show()


# # Plotting two images to compare
# steps = 100
# fig, axarr, plot_next = image_map_factory(1,2, hspace=0.1);
# lat_range, lon_range = REGION_COORDS['nwus']
# lats = np.linspace(lat_range[0], lat_range[1], steps)
# lons = np.linspace(lon_range[0], lon_range[1], steps)
# (hr_img, lr_img, prev_hr_img) = myData.__getitem__(10)
# plot_next(axarr[0], lr_img.numpy(), lats, lons, min_max=[0,60], title='Low Resolution Daily Precipitation')
# plot_next(axarr[1], hr_img.numpy(), lats, lons, min_max=[0,60], title='High Resolution Daily Precipitation')
# plt.show()