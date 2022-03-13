import os
import torch
from torch.utils import data
from generator import *
from dataloader import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# just generating latent features
def get_gen_embeddings(network, latent, input2):
    latent = torch.cat([latent, input2], 1)
    latent = network.project1(latent)
    latent = network.project2(latent)
    latent, _ = network.att(latent)
    latent = network.up_project1(latent)
    latent = network.up_project2(latent)
    return latent


device=torch.device('cpu')
style_dim=512
regions = ['nwus']
positions = [(0,0),(0,39),(39,0),(20,20),(39,39)]
# positions = [(x,y) for x in range(0,40,5) for y in range(0,40,5)]

# grab a saved GAN generator
network=Generator(size1=40,size2=40,style_dim=style_dim,coord_size=3)
network.load_state_dict(torch.load('generator10_.pt', map_location=device))

# getting embeddings for each image pair
res_folder='results/'
mode='ours/'
for test_set in ['test']:  
    test_data=EraiCpcDataset('./tensordata-precip-40',test_set, regions=regions)
    test_loader=torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=True,
                                            num_workers=0,pin_memory=True,sampler=None,
                                            drop_last=True)
    network=network.to(device)
    network=network.eval()
    # storing features for a particular coordinate here
    n_samples = len(test_data)
    n_pos = len(positions)
    point_features = np.zeros((n_pos, n_samples,style_dim))
    point_values = np.zeros((n_pos, n_samples))
    patch_mean = np.zeros((n_pos, n_samples))
    with torch.no_grad():
        for batch_num,data in enumerate(test_loader):
            hr_img,lr_img,prev_hr_img,coord,name=data['hr_img'].to(device),data['lr_img'].to(device),data['prev_hr_img'].to(device),data['coord'].to(device),data['name']
            latent = get_gen_embeddings(network, lr_img, prev_hr_img)
            for pos_num, pos in enumerate(positions):
                point_features[pos_num,batch_num,:] = latent.detach().numpy()[0,:,pos[0],pos[1]]
                point_values[pos_num,batch_num] = lr_img.detach().numpy()[0,0,pos[0],pos[1]]
                patch_mean[pos_num,batch_num] = np.mean(lr_img.detach().numpy())
            if(batch_num%30==0):
                print(f'{batch_num}/{n_samples}')

# calculating historical avg precipitation for each coord
point_mean = np.mean(point_values, axis=1)
print(point_mean.shape)
hist_labels = np.repeat(point_mean, n_samples)
print(hist_labels.shape)
point_dist = [np.linalg.norm(np.array([0,0])-np.array(pos)) for pos in positions]
dist_labels = np.repeat(point_dist, n_samples)

# Standardizing features and performing PCA
point_values = np.reshape(point_values, (n_samples*n_pos,)) + 0.0001
patch_mean= np.reshape(patch_mean, (n_samples*n_pos,)) + 0.0001
point_features = np.reshape(point_features, (n_samples*n_pos, style_dim))
point_features = StandardScaler().fit_transform(point_features)
pca = PCA(n_components=2)
pc = pca.fit_transform(point_features)

# projection of features and precipitation
plt.scatter(pc[:,0], pc[:,1], c=np.log(point_values), cmap='jet')
clb = plt.colorbar()
clb.ax.set_title('log precipitation',fontsize=8)
plt.xlabel('principle component 1')
plt.ylabel('principle component 2')
plt.show()
plt.close()

# projection of features and regional mean precipitation
plt.scatter(pc[:,0], pc[:,1], c=np.log(patch_mean), cmap='jet')
clb = plt.colorbar()
clb.ax.set_title('log mean precipitation of nwus',fontsize=8)
plt.xlabel('principle component 1')
plt.ylabel('principle component 2')
plt.show()
plt.close()

# projection of features and historical precipitation
plt.scatter(pc[:,0], pc[:,1], c=hist_labels, cmap='rainbow')
clb = plt.colorbar()
clb.ax.set_title('historical average daily rainfall',fontsize=8)
plt.xlabel('principle component 1')
plt.ylabel('principle component 2')
plt.show()
plt.close()

# projection of features and distance from top right
plt.scatter(pc[:,0], pc[:,1], c=dist_labels, cmap='rainbow')
clb = plt.colorbar()
clb.ax.set_title('distance from top right corner',fontsize=8)
plt.xlabel('principle component 1')
plt.ylabel('principle component 2')
plt.show()
plt.close()

