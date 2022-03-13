import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from plot import *
from statsmodels.graphics.tsaplots import plot_acf


REGION_COORDS = {'nwus':([38,48],[238,248]), 'seus':([28,38], [268,278]), 
				 'neus':([35,45], [273,283]), 'swus':([32,42], [242,252]),
				 'mwus':([35,45],[255, 265])}
step=40


data =  EraiCpcDataset('./tensordata-precip-40', regions=['nwus'])

lr_precip = np.zeros((data.__len__(),step**2))
mr_precip = np.zeros((data.__len__(),step**2))
hr_precip = np.zeros((data.__len__(),step**2))

for i in range(data.__len__()):
	curr_precip_data = data.__getitem__(i)
	lr_precip[i,:] = torch.reshape(curr_precip_data['lr_img'], (step**2,)).numpy()
	mr_precip[i,:] = torch.reshape(curr_precip_data['hr_img'], (step**2,)).numpy()

print(mr_precip.shape)
print(mr_precip[:,0].shape)
plot_acf(mr_precip[:,0], lags=400)
plt.title('Autocorrelation of Precipitation Data (1 Pixel, 400 days)')
plt.show()