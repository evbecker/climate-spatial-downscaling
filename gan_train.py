
from torch.utils import data
from generator import *
from evaluation_metrics import *

from torch.utils.data import Subset
from dataloader import EraiCpcDataset
from networks_ import AE
import random


epochs=100
lr=0.0001
l1_lambda=1
device='cuda:1'


def validate(network,dataloader,criterion,mode):
    network.eval()
    n=0
    total_loss=0
    for batch_num,data in enumerate(dataloader):
        hr_img,lr_img,prev_hr_img,coord=data['hr_img'].to(device),data['lr_img'].to(device),data['prev_hr_img'].to(device),data['coord'].to(device)
        if mode!='ours':
            coord=coord[:,:3,:,:]
        noise=mixing_noise(1,style_dim,1,device)
        fake_hr=network_g(coord,lr_img,prev_hr_img,noise)
        loss=criterion(fake_hr,hr_img).mean()
        total_loss+=loss.item()
        n+=1
    return total_loss/n

def test(network,dataloader,mode):
    network.eval()
    rmse=[]
    pc=[]
    for batch_num,data in enumerate(dataloader):
        hr_img,lr_img,prev_hr_img,coord=data['hr_img'].to(device),data['lr_img'].to(device),data['prev_hr_img'].to(device),data['coord'].to(device)
        if mode!='ours':
            coord=coord[:,:3,:,:]
        noise=mixing_noise(1,style_dim,1,device)
        pred=network_g(coord,lr_img,prev_hr_img,noise)
        pred=pred.detach().cpu().numpy()
        hr_img=hr_img.detach().cpu().numpy()
        rmse.append(RMSE(pred,hr_img))
        pc.append(PCorrelation(pred,hr_img))
    return np.mean(rmse),np.std(rmse),np.mean(pc),np.std(pc)

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

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def g_rec_loss(fake, real):
    loss = F.l1_loss(fake, real)
#     loss = F.mse_loss(fake, real, reduction='sum') / fake.size(0)

    return loss
for l1_lambda in [0.01,0.05,0.1,0.2,0.5,1,5,10]:
#for l1_lambda in [0.2]:
    for mode in ['ours','ead']:
        batch_size=8
        style_dim=512
        train_data=EraiCpcDataset('./tensordata','train')
        val_data=EraiCpcDataset('./tensordata','val')
        test_data=EraiCpcDataset('./tensordata','test')
        train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
        val_loader=torch.utils.data.DataLoader(val_data,batch_size=1,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
        test_loader=torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
        if mode=='ours':
            network_g=Generator(size1=40,size2=40,style_dim=style_dim,coord_size=4)
        else:
            network_g=Generator(size1=40,size2=40,style_dim=style_dim,coord_size=3)
        #network=AE(input_channels=1,num_layers=3,base_num=16)
        network_g=network_g.to(device)
        network_d=Discriminator(size1=40,size2=40)
        network_d=network_d.to(device)
        criterion_g=torch.nn.MSELoss()
        optimizer_g=torch.optim.Adam(network_g.parameters(),lr=lr)
        optimizer_d=torch.optim.Adam(network_d.parameters(),lr=lr)
        loss_max=10000000

        for epoch in range(0,epochs):
            for batch,data in enumerate(train_loader):
                optimizer_g.zero_grad()
                optimizer_d.zero_grad()
                network_g.train()
                network_d.train()
                hr_img,lr_img,prev_hr_img,coord=data['hr_img'].to(device),data['lr_img'].to(device),data['prev_hr_img'].to(device),data['coord'].to(device)
                if mode!='ours':
                    coord=coord[:,:3,:,:]
                requires_grad(network_g,False)
                requires_grad(network_d,True)
                noise=mixing_noise(batch_size,style_dim,1,device)
                fake_hr=network_g(coord,lr_img,prev_hr_img,noise)
                fake_input_d=torch.cat((fake_hr,lr_img,prev_hr_img),1)
                fake_pred=network_d(fake_input_d)
                real_input_d=torch.cat((hr_img,lr_img,prev_hr_img),1)
                real_pred=network_d(real_input_d)
                d_loss=d_logistic_loss(real_pred,fake_pred)
                d_loss.backward()
                optimizer_d.step()


                requires_grad(network_g,True)
                requires_grad(network_d,False)
                noise=mixing_noise(batch_size,style_dim,1,device)
                fake_hr=network_g(coord,lr_img,prev_hr_img,noise)
                fake_input_g=torch.cat((fake_hr,lr_img,prev_hr_img),1)
                fake_pred=network_d(fake_input_g)
                g_gan_loss=g_nonsaturating_loss(fake_pred)
                rec_loss=g_rec_loss(fake_hr,hr_img)
                g_loss=g_gan_loss+l1_lambda*rec_loss

                network_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()


            with torch.no_grad():
                total_loss=validate(network_g,val_loader,criterion_g,mode)
                rmse_m,rmse_s,pc_m,pc_s=test(network_g,test_loader,mode)
                if total_loss<loss_max:
                    loss_max=total_loss
                    if mode=='ours':
                        torch.save(network_g.state_dict(),'generator'+str(l1_lambda)+'.pt')
                    else:
                        torch.save(network_g.state_dict(),'generator'+str(l1_lambda)+'_.pt')
                    best=[rmse_m,rmse_s,pc_m,pc_s]
                print(epoch,g_loss.item(),total_loss,rmse_m,rmse_s,pc_m,pc_s)
        print(mode,l1_lambda,best)



