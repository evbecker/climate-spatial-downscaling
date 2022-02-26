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

    def __init__(self, img_path):
        #self.csv_path = csv_path
        self.img_path = img_path
        # Read the csv file
        #self.data_info = pd.read_csv(csv_path, header=0)
        # Get list of image names from csv
        #self.erai = np.asarray(self.data_info.iloc[0])
        self.erai=[]
        self.cpc=[]
        date='2000-01-01'
        while os.path.exists(img_path+'/erai-nwus-precip-'+date+'.pt') and os.path.exists(img_path+'/cpc-nwus-precip-'+date+'.pt'):
            self.erai.append('erai-nwus-precip-'+date+'.pt')
            self.cpc.append('cpc-nwus-precip-'+date+'.pt')
            date=next_date(date)

        #self.cpc = np.asarray(self.data_info.iloc[1])
        # length is number of time samples
        self.data_len = len(self.erai)
        self.elevation=torch.load(img_path+'/nwus-elevation-40x40.pt')

    def __getitem__(self, index):
        # for index corresponding to time t, returns high and low
        # resolution images for that time, as well as high res
        # at the previous timestep
        lr_img = torch.load(os.path.join(self.img_path, self.erai[index])).unsqueeze(0)
        hr_img = torch.load(os.path.join(self.img_path, self.cpc[index])).unsqueeze(0)
        prev_index = (index - 1)%self.data_len
        prev_hr_img = torch.load(os.path.join(self.img_path, self.cpc[prev_index])).unsqueeze(0)
        coord=torch.zeros((4,lr_img.size(1),lr_img.size(2)))
        coord[3,:,:]=date_converter(self.erai[index][-13:-3])
        lat=torch.linspace(38,48,lr_img.size(0)).unsqueeze(1)
        lon=torch.linspace(238,248,lr_img.size(1)).unsqueeze(0)
        coord[0,:,:]=lat
        coord[1,:,:]=lon
        coord[2,:,:]=self.elevation
        #print(self.erai[index][-13:-3])
        return {'hr_img':hr_img,'lr_img':lr_img,'prev_hr_img':prev_hr_img,'coord':coord}

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    # just testing that we return three tensor images for a given index
    #print(torch.load('./tensordata/erai-nwus-precip-2000-01-02.pt'))
    myData =  EraiCpcDataset('./tensordata')
    print(myData.__getitem__(9)[0].size())
