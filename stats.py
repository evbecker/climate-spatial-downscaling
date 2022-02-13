import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *

"""
Comparing spatial averages of precipitation in the Northwest US
for the year 2000

"""

myData =  EraiCpcDataset('./tensordata/nwus-2000-01-01.csv', './tensordata')

lr_avg_precip = np.zeros(myData.__len__())
hr_avg_precip = np.zeros(myData.__len__())

for i in range(myData.__len__()):
	(hr_img, lr_img, prev_hr_img) = myData.__getitem__(i)
	lr_avg_precip[i] = torch.mean(lr_img).numpy()
	hr_avg_precip[i] = torch.mean(hr_img).numpy()


plt.plot(list(range(myData.__len__())), lr_avg_precip, label="low resolution")
plt.plot(list(range(myData.__len__())), hr_avg_precip, label="high resolution")
plt.xlabel("day of the year (2000)")
plt.ylabel("average precipitation (mm) for NW US")
plt.legend()
plt.show()
