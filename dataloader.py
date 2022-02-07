import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


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
        lr_img = torch.load(os.path.join(self.img_path, self.erai[index]))
        hr_img = torch.load(os.path.join(self.img_path, self.cpc[index]))
        prev_index = (index - 1)%self.data_len
        prev_hr_img = torch.load(os.path.join(self.img_path, self.cpc[prev_index]))

        return (hr_img, lr_img, prev_hr_img)

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    # just testing that we return three tensor images for a given index
    myData =  EraiCpcDataset('./tensordata/nwus-2000-01-01.csv', './tensordata')
    print(myData.__getitem__(10))
