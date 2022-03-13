import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import csv

def date_converter(s,base_year=2000,base_month=1,base_day=1):
    days_per_month={
        1:31,
        2:28,
        3:31,
        4:30,
        5:31,
        6:30,
        7:31,
        8:31,
        9:30,
        10:31,
        11:30,
        12:31,
    }
    year=int(s[:4])
    if year%4==0:
        days_per_month[2]=29
    t=(year-base_year)*365
    for i in range(base_year,year):
        if i%4==0:
            t+=1
    month=int(s[5:7])
    for i in range(base_month,month):
        t+=days_per_month[i]
    day=int(s[8:10])
    t+=day-base_day
    return t

def next_date(date):
    days_per_month={
        1:31,
        2:28,
        3:31,
        4:30,
        5:31,
        6:30,
        7:31,
        8:31,
        9:30,
        10:31,
        11:30,
        12:31,
    }
    year=int(date[:4])
    month=int(date[5:7])
    day=int(date[8:10])
    if year%4==0:
        days_per_month[2]=29
    if day==days_per_month[month]:
        day=1
        if month!=12:
            month+=1
        else:
            month=1
            year+=1
    else:
        day+=1
    year=str(year)
    if month<10:
        month='0'+str(month)
    else:
        month=str(month)
    if day<10:
        day='0'+str(day)
    else:
        day=str(day)
    return str(year)+'-'+str(month)+'-'+str(day)



class EraiCpcDataset(Dataset):

    def __init__(self, img_path, mode='train', regions=['neus','nwus','seus','swus']):
        #self.csv_path = csv_path
        self.img_path = img_path
        # Read the csv file
        #self.data_info = pd.read_csv(csv_path, header=0)
        # Get list of image names from csv
        #self.erai = np.asarray(self.data_info.iloc[0])
        self.erai=[]
        self.cpc=[]
        self.regions=[]
        self.elevation={}
        self.latitude={}
        self.longitude={}
        if mode!='new_test':
            for r in regions:
                if mode=='train':
                    date='2002-01-01'
                    end_date='3000-01-01'
                elif mode=='val':
                    date='2001-01-01'
                    end_date='2002-01-01'
                elif mode=='test':
                    date='2000-01-01'
                    end_date='2001-01-01'
                while os.path.exists(img_path+'/erai-'+r+'-precip-'+date+'.pt') and os.path.exists(img_path+'/cpc-'+r+'-precip-'+date+'.pt') and date!=end_date:
                    self.erai.append('erai-'+r+'-precip-'+date+'.pt')
                    self.cpc.append('cpc-'+r+'-precip-'+date+'.pt')
                    self.regions.append(r)
                    date=next_date(date)
                self.elevation[r]=torch.load(img_path+'/'+r+'-elevation-40x40.pt')
            self.latitude['nwus']=torch.linspace(38,48,40).unsqueeze(1)
            self.longitude['nwus']=torch.linspace(238,248,40).unsqueeze(0)
            self.latitude['seus']=torch.linspace(28,38,40).unsqueeze(1)
            self.longitude['seus']=torch.linspace(268,278,40).unsqueeze(0)
            self.latitude['neus']=torch.linspace(35,45,40).unsqueeze(1)
            self.longitude['neus']=torch.linspace(273,283,40).unsqueeze(0)
            self.latitude['swus']=torch.linspace(32,42,40).unsqueeze(1)
            self.longitude['swus']=torch.linspace(242,252,40).unsqueeze(0)
        else:
            date='2000-01-01'
            end_date='3000-01-01'
            r='mwus'
            while os.path.exists(img_path+'/erai-'+r+'-precip-'+date+'.pt') and os.path.exists(img_path+'/cpc-'+r+'-precip-'+date+'.pt') and date!=end_date:
                self.erai.append('erai-'+r+'-precip-'+date+'.pt')
                self.cpc.append('cpc-'+r+'-precip-'+date+'.pt')
                self.regions.append(r)
                date=next_date(date)
            self.elevation[r]=torch.load(img_path+'/'+r+'-elevation-40x40.pt')
            self.latitude['mwus']=torch.linspace(35,45,40).unsqueeze(1)
            self.longitude['mwus']=torch.linspace(255,265,40).unsqueeze(0)



        #self.cpc = np.asarray(self.data_info.iloc[1])
        # length is number of time samples
        self.data_len = len(self.erai)

    def __getitem__(self, index):
        # for index corresponding to time t, returns high and low
        # resolution images for that time, as well as high res
        # at the previous timestep
        lr_img = torch.load(os.path.join(self.img_path, self.erai[index])).unsqueeze(0)
        hr_img = torch.load(os.path.join(self.img_path, self.cpc[index])).unsqueeze(0)
        prev_index = (index - 1)%self.data_len
        prev_hr_img = torch.load(os.path.join(self.img_path, self.cpc[prev_index])).unsqueeze(0)
        coord=torch.zeros((4,lr_img.size(1),lr_img.size(2)))
        r=self.regions[index]
        coord[0,:,:]=date_converter(self.erai[index][-13:-3])
        coord[1,:,:]=self.latitude[r]
        coord[2,:,:]=self.longitude[r]
        coord[3,:,:]=self.elevation[r]
        #print(self.erai[index][-13:-3])
        return {'hr_img':hr_img,'lr_img':lr_img,'prev_hr_img':prev_hr_img,'coord':coord,'name':self.cpc[index],'region':r}

    def __len__(self):
        return self.data_len





class EraiCpcWrfDataset(Dataset):

    def __init__(self, img_path, mode='train', regions=['neus','nwus','seus','swus']):
        #self.csv_path = csv_path
        self.img_path = img_path
        # Read the csv file
        #self.data_info = pd.read_csv(csv_path, header=0)
        # Get list of image names from csv
        #self.erai = np.asarray(self.data_info.iloc[0])
        self.erai=[]
        self.cpc=[]
        self.wrf=[]
        self.regions=[]
        self.elevation={}
        self.latitude={}
        self.longitude={}
        if mode!='new_test':
            for r in regions:
                if mode=='train':
                    date='2003-01-01'
                    end_date='3000-01-01'
                elif mode=='val':
                    date='2002-01-01'
                    end_date='2003-01-01'
                elif mode=='test':
                    date='2001-01-01'
                    end_date='2002-01-01'
                while os.path.exists(img_path+'/erai-'+r+'-precip-'+date+'.pt') and os.path.exists(img_path+'/cpc-'+r+'-precip-'+date+'.pt') and os.path.exists(img_path+'/wrf-'+r+'-precip-'+date+'.pt') and date!=end_date:
                    self.erai.append('erai-'+r+'-precip-'+date+'.pt')
                    self.cpc.append('cpc-'+r+'-precip-'+date+'.pt')
                    self.wrf.append('wrf-'+r+'-precip-'+date+'.pt')
                    self.regions.append(r)
                    date=next_date(date)
                self.elevation[r]=torch.load(img_path+'/'+r+'-elevation-160x160.pt')
            self.latitude['nwus']=torch.linspace(38,48,160).unsqueeze(1)
            self.longitude['nwus']=torch.linspace(238,248,160).unsqueeze(0)
            self.latitude['seus']=torch.linspace(28,38,160).unsqueeze(1)
            self.longitude['seus']=torch.linspace(268,278,160).unsqueeze(0)
            self.latitude['neus']=torch.linspace(35,45,160).unsqueeze(1)
            self.longitude['neus']=torch.linspace(273,283,160).unsqueeze(0)
            self.latitude['swus']=torch.linspace(32,42,160).unsqueeze(1)
            self.longitude['swus']=torch.linspace(242,252,160).unsqueeze(0)
        else:
            date='2001-01-01'
            end_date='3000-01-01'
            r='mwus'
            #print(os.path.exists(img_path+'/erai-'+r+'-precip-'+date+'.pt'),os.path.exists(img_path+'/cpc-'+r+'-precip-'+date+'.pt'),os.path.exists(img_path+'/wrf-'+r+'-precip-'+date+'.pt'))
            while os.path.exists(img_path+'/erai-'+r+'-precip-'+date+'.pt') and os.path.exists(img_path+'/cpc-'+r+'-precip-'+date+'.pt') and os.path.exists(img_path+'/wrf-'+r+'-precip-'+date+'.pt') and date!=end_date:
                self.erai.append('erai-'+r+'-precip-'+date+'.pt')
                self.cpc.append('cpc-'+r+'-precip-'+date+'.pt')
                self.wrf.append('wrf-'+r+'-precip-'+date+'.pt')
                self.regions.append(r)
                date=next_date(date)
            self.elevation[r]=torch.load(img_path+'/'+r+'-elevation-160x160.pt')
            self.latitude['mwus']=torch.linspace(35,45,160).unsqueeze(1)
            self.longitude['mwus']=torch.linspace(255,265,160).unsqueeze(0)



        #self.cpc = np.asarray(self.data_info.iloc[1])
        # length is number of time samples
        self.data_len = len(self.erai)

    def __getitem__(self, index):
        # for index corresponding to time t, returns high and low
        # resolution images for that time, as well as high res
        # at the previous timestep
        slr_img = torch.load(os.path.join(self.img_path, self.erai[index])).unsqueeze(0)
        lr_img = torch.load(os.path.join(self.img_path, self.cpc[index])).unsqueeze(0)
        hr_img=torch.load(os.path.join(self.img_path, self.wrf[index])).unsqueeze(0)
        prev_index = (index - 1)%self.data_len
        prev_hr_img = torch.load(os.path.join(self.img_path, self.wrf[prev_index])).unsqueeze(0)
        coord=torch.zeros((4,lr_img.size(1),lr_img.size(2)))
        r=self.regions[index]
        coord[0,:,:]=date_converter(self.erai[index][-13:-3])
        coord[1,:,:]=self.latitude[r]
        coord[2,:,:]=self.longitude[r]
        coord[3,:,:]=self.elevation[r]
        #print(self.erai[index][-13:-3])
        return {'hr_img':hr_img.type(torch.FloatTensor),'lr_img':lr_img.type(torch.FloatTensor),'prev_hr_img':prev_hr_img.type(torch.FloatTensor),'coord':coord,'name':self.cpc[index],'region':r,'slr_img':slr_img.type(torch.FloatTensor)}

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    # just testing that we return three tensor images for a given index
    #print(torch.load('./tensordata/erai-nwus-precip-2000-01-02.pt'))
    myData =  EraiCpcWrfDataset('./tensordata-precip-160','new_test')
    print(len(myData))
    print(myData.__getitem__(0)['name'])
