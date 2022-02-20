import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

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



class EraiCpcDataset(Dataset):

    def __init__(self, csv_path, img_path):
        self.csv_path = csv_path
        self.img_path = img_path
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # Get list of image names from csv
        self.erai = np.asarray(self.data_info.iloc[0])
        self.cpc = np.asarray(self.data_info.iloc[1])
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
        coord=torch.zeros((3,lr_img.size(1),lr_img.size(2)))
        coord[2,:,:]=date_converter(self.erai[index][-13:-3])
        lat=torch.linspace(38,48,lr_img.size(0)).unsqueeze(1)
        lon=torch.linspace(238,248,lr_img.size(1)).unsqueeze(0)
        coord[0,:,:]=lat
        coord[1,:,:]=lon
        #print(self.erai[index][-13:-3])
        return {'hr_img':hr_img,'lr_img':lr_img,'prev_hr_img':prev_hr_img,'coord':coord}

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    # just testing that we return three tensor images for a given index
    #print(torch.load('./tensordata/erai-nwus-precip-2000-01-02.pt'))
    myData =  EraiCpcDataset('./tensordata/nwus-2000-01-01.csv', './tensordata')
    print(myData.__getitem__(9)[0].size())
