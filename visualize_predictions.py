import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from plot import *

MODELS = ['AE', 'EAD', 'naive', 'ours']
REGION_COORDS = {'nwus':([38,48],[238,248]), 'seus':([28,38], [268,278]), 
				 'neus':([35,45], [273,283]), 'swus':([32,42], [242,252]),
				 'mwus':([35,45],[255, 265])}

# choosing which results to show
model_path='./results'
data_path='./tensordata-precip-40'
date='2000-01-20'
region='mwus'
steps = 40

# coordinates for chosen region
lat_range, lon_range = REGION_COORDS[region]
lats = np.linspace(lat_range[0], lat_range[1], steps)
lons = np.linspace(lon_range[0], lon_range[1], steps)

# getting saved model predictions
predictions = []
max_value = 0
for i, model in enumerate(MODELS):
	img_path = os.path.join(model_path, model, f'cpc-{region}-precip-{date}.pt')
	# img_path = os.path.join(model_path, model, f'cpc-nwus-precip-2000-09-08.pt')
	print(img_path)
	predictions.append(torch.load(img_path, map_location=torch.device('cpu')))
	if torch.max(predictions[i]) > max_value:
		max_value = torch.max(predictions[i])

# getting actual data
erai_path = os.path.join(data_path, f'erai-{region}-precip-{date}.pt')
erai_precip = torch.load(erai_path, map_location=torch.device('cpu'))
cpc_path = os.path.join(data_path, f'cpc-{region}-precip-{date}.pt')
cpc_precip = torch.load(cpc_path, map_location=torch.device('cpu'))

# plotting on US map
fig, axarr, plot_next = image_map_factory(3,2, hspace=0.15, wspace=0.15, cbar_per_subplot=True, 
										  gridlines=False, cbar_orientation='vertical')

plt.suptitle(f'Daily Precipitation (MM) Midwest {date}', fontsize=18)

plot_next(axarr[0,0], erai_precip.numpy(), lats, lons, 
			 min_max=[0, max_value], title= '(INPUT) ERA Low Resolution')

plot_next(axarr[0,1], cpc_precip.numpy(), lats, lons, 
			 min_max=[0, max_value], title='(TRUE) CPC High Resolution')			

plot_next(axarr[1,0], predictions[0].numpy(), lats, lons, 
			 min_max=[0, max_value], title='(PREDICTED) AE High Resolution')

plot_next(axarr[1,1], predictions[1].numpy(), lats, lons, 
			 min_max=[0, max_value], title='(PREDICTED) EAD High Resolution')

plot_next(axarr[2,0], predictions[2].numpy(), lats, lons, 
			 min_max=[0, max_value], title='(PREDICTED) Naive High Resolution')

plot_next(axarr[2,1], predictions[3].numpy(), lats, lons,  
			 min_max=[0, max_value], title='(PREDICTED) Our Model High Resolution')

plt.show()