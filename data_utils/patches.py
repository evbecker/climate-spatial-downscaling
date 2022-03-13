import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from plot import *

"""
Visualizing chosen coordinate patches across the continental united states
"""

REGION_COORDS = {'nwus':([38,48],[238,248]), 'seus':([28,38], [268,278]), 
                 'neus':([35,45], [273,283]), 'swus':([32,42], [242,252]),
                 'mwus':([35,45],[255, 265])}
colors = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']

cmap="viridis"
figsize=(6,4)
min_max=None  
bins=100
steps=40

proj = ccrs.PlateCarree()
fig = plt.figure(figsize=figsize)
ax = plt.axes(projection=proj)
ax.set_extent([238, 278, 28, 48], proj)
min_val, max_val = (0, 100)
breaks = np.linspace(min_val-1.0E-15, max_val+1.0E-15, bins+1)

color_i = 0
for region, (lat_range, lon_range) in REGION_COORDS.items():
    lat_range, lon_range = REGION_COORDS[region]
    lats = np.linspace(lat_range[0], lat_range[1], steps)
    lons = np.linspace(lon_range[0], lon_range[1], steps)
    values = np.zeros((steps,steps))
    values[0,:] =100
    values[steps-1,:] =100
    values[:,0] =100
    values[:,steps-1] =100

    # plot_image_map(values, y, x)
    # plt.show()
    
    ax.contourf(lons, lats, values, breaks, cmap=colors[color_i], transform=proj, extend='max')
    color_i +=1

ax = plt.gca()
ax.coastlines()
plt.show()