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
				 'neus':([35,45], [273,283]), 'swus':([32,42], [242,252]),
				 'mwus':([35,45],[255, 265])}

data =  EraiCpcWrfDataset('./tensordata-precip-160', regions=['neus'])


lr_avg_precip = np.zeros(data.__len__())
mr_avg_precip = np.zeros(data.__len__())
hr_avg_precip = np.zeros(data.__len__())
lr_max_precip = np.zeros(data.__len__())
mr_max_precip = np.zeros(data.__len__())
hr_max_precip = np.zeros(data.__len__())

for i in range(data.__len__()):
	curr_precip_data = data.__getitem__(i)
	lr_avg_precip[i] = torch.mean(curr_precip_data['lr_img']).numpy()
	mr_avg_precip[i] = torch.mean(curr_precip_data['hr_img']).numpy()
	lr_max_precip[i] = torch.max(curr_precip_data['lr_img']).numpy()
	mr_max_precip[i] = torch.max(curr_precip_data['hr_img']).numpy()

for i in range(ERA_WRF_data.__len__()):
	curr_precip_data = ERA_WRF_data.__getitem__(i)
	hr_avg_precip[i] = torch.mean(curr_precip_data['hr_img']).numpy()
	hr_max_precip[i] = torch.max(curr_precip_data['hr_img']).numpy()


# comparing spatial averages over time
end=100
plt.plot(list(range(ERA_CPC_data.__len__()))[:end], lr_avg_precip[:end], label="ERA low resolution")
plt.plot(list(range(ERA_CPC_data.__len__()))[:end], mr_avg_precip[:end], label="CPC medium resolution")
plt.plot(list(range(ERA_WRF_data.__len__()))[:end], hr_avg_precip[:end], label="WRF high resolution")
plt.xlabel("days since 10/01/2000")
plt.ylabel("AVERAGE precipitation (mm) for NW US")
plt.legend()
plt.show()

# comparing spatial max over time
plt.plot(list(range(ERA_CPC_data.__len__()))[:end], lr_max_precip[:end], label="ERA low resolution")
plt.plot(list(range(ERA_CPC_data.__len__()))[:end], mr_max_precip[:end], label="CPC medium resolution")
plt.plot(list(range(ERA_WRF_data.__len__()))[:end], hr_max_precip[:end], label="WRF high resolution")
plt.xlabel("days since 10/01/2000")
plt.ylabel("MAX precipitation (mm) for NW US")
plt.legend()

plt.show()