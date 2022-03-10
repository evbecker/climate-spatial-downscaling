from torch.utils import data
from generator import *
from evaluation_metrics import *
from autoencoder import AutoEncoder

from torch.utils.data import Subset
from dataloader import EraiCpcDataset,EraiCpcWrfDataset
from networks_ import AE
import random
import os

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

epochs=100
lr=0.0001
l1_lambda=1
device='cuda:0'


#for l1_lambda in [0.01,0.05,0.1,0.2,0.5,1,5,10]:
res_folder='results_160/'
reso=160
if not os.path.exists(res_folder):
    os.mkdir(res_folder)
for mode in ['ours/']:
    for test_set in ['test','new_test']:
        if not os.path.exists(res_folder+mode):
            os.mkdir(res_folder+mode)
        style_dim=512
        if reso==40:
            test_data=EraiCpcDataset('./tensordata',test_set)
        else:
            test_data=EraiCpcWrfDataset('./tensordata-precip-160',test_set)
        test_loader=torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
        if  mode=='ours/':
            network=Generator(size1=reso,size2=reso,style_dim=style_dim,coord_size=4)
            network.load_state_dict(torch.load('_generator10.pt'))
        elif mode=='EAD/':
            network=Generator(size1=reso,size2=reso,style_dim=style_dim,coord_size=3)
            network.load_state_dict(torch.load('generator10_.pt'))
        elif mode=='naive/':
            network=AutoEncoder(size1=reso,size2=reso)
            network.load_state_dict(torch.load('autoencoder_naive.pt'))
        elif mode=='AE/':
            network=AE(input_channels=1,num_layers=3,base_num=16)
            network.load_state_dict(torch.load('autoencoder_ae.pt'))
        network=network.to(device)
        network=network.eval()
        with torch.no_grad():
            for batch_num,data in enumerate(test_loader):
                hr_img,lr_img,prev_hr_img,coord,name=data['hr_img'].to(device),data['lr_img'].to(device),data['prev_hr_img'].to(device),data['coord'].to(device),data['name']
                if mode=='EAD/':
                    coord=coord[:,:3,:,:]
                noise=mixing_noise(1,style_dim,1,device)
                if mode=='ours/' or mode=='EAD/':
                    pred=network(coord,lr_img,prev_hr_img,noise)
                elif mode=='naive/' or mode=='AE/':
                    pred=network(lr_img)
                torch.save(pred.detach(),res_folder+mode+name[0])

