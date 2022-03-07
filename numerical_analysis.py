from torch.utils import data
from generator import *
from evaluation_metrics import *
from autoencoder import AutoEncoder

from torch.utils.data import Subset
from dataloader import EraiCpcDataset
from networks_ import AE
import random
import os
from evaluation_metrics import *
device='cuda'

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




def evaluate(mode,model_path):
    style_dim=512
    if  mode=='ours/':
        network=Generator(size1=40,size2=40,style_dim=style_dim,coord_size=4)
        network.load_state_dict(torch.load(model_path))
    elif mode=='EAD/':
        network=Generator(size1=40,size2=40,style_dim=style_dim,coord_size=3)
        network.load_state_dict(torch.load(model_path))
    elif mode=='naive/':
        network=AutoEncoder(size1=40,size2=40)
        network.load_state_dict(torch.load(model_path))
    elif mode=='AE/':
        network=AE(input_channels=1,num_layers=3,base_num=16)
        network.load_state_dict(torch.load(model_path))
    for test_set in ['test','test_new']:
        test_data=EraiCpcDataset('./tensordata',test_set)
        test_loader=torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
        network=network.to(device)
        network.eval()
        rmse=[]
        pc=[]
        if test_set=='test':
            dic_pred,dic_target,regions_dic=max_per_init()
        elif test_set=='test_new':
            dic_pred,dic_target,regions_dic=max_per_init(regions=['mwus'],years=7)
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
                pred=pred.detach().cpu().numpy()
                hr_img=hr_img.detach().cpu().numpy()
                rmse.append(RMSE(pred,hr_img))
                pc.append(PCorrelation(pred,hr_img))
                dic_pred,dic_target=max_per_calculate(pred,hr_img,name,dic_pred,dic_target,regions_dic,2000)
        mean_dif,std_dif=max_per(dic_pred,dic_target)
        print(test_set,model_path,np.mean(rmse),np.std(rmse),np.mean(pc),np.std(pc),mean_dif,std_dif)
for i in [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]:
    evaluate('ours','generator'+str(i)+'.pt')
    evaluate('EAD','generator'+str(i)+'_.pt')
evaluate('AE','autoencoder_ae.pt')
evaluate('naive','autoencoder_naive.pt')



