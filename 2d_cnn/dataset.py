import scipy.io as sio
import numpy as np
import torch
import os
import torch.utils.data
from torchvision import transforms
from sklearn.preprocessing import normalize

def max_(x, y):
    x = x.reshape(-1, 1)
    x = x.repeat(300, axis=1)
    y = y.reshape(1, -1)
    y = y.repeat(600, axis=0)
    x[(x-y)>0] = 0
    y[(x-y)<0] = 0
    res = x + y
    res = res[np.newaxis, :, :]
    return torch.from_numpy(res)

def average_(x, y):
    x = x.reshape(1, -1)
    # x = normalize(x, norm='l2')
    x = x.reshape(-1, 1)
    x = x.repeat(300, axis=1)

    y = y.reshape(1, -1)
    # y = normalize(y, norm='l2')
    y = y.repeat(600, axis=0)

    res = (x + y) / 2
    res = normalize(res, norm='l2') # 19/11/1 add
    res = res[np.newaxis, :, :]
    return torch.from_numpy(res)

def sum_(x, y):
    x = x.reshape(-1, 1)
    x = x.repeat(300, axis=1)
    y = y.reshape(1, -1)
    y = y.repeat(600, axis=0)
    res = x + y
    res = res[np.newaxis, :, :]
    return torch.from_numpy(res)

def outProduct_(x, y):
    res = np.outer(x, y, out=None)
    res = res[np.newaxis, :, :]
    return torch.from_numpy(res)

def read_mat_3D(mat_path):
    data = sio.loadmat(mat_path)
    m = np.array(data['matrix'])
    m = torch.from_numpy(m)
    m = m.permute(2, 1, 0)
    return m

def read_mat_2D(mat_path):
    data = sio.loadmat(mat_path)
    m = np.array(data['matrix'])
    m = torch.from_numpy(m)
    m = m[np.newaxis, :, :]
    return m

def read_mat_1D(mat_path):
    data = sio.loadmat(mat_path)
    m = np.array(data['matrix'])
    # m = torch.from_numpy(m)
    # m = m[np.newaxis, :, :]
    return m
    

class MyDataset(torch.utils.data.Dataset):

    
    def __init__(self, NN_path, NoneNN_path, file_path, loader=read_mat_1D, transform=None):

        with open(file_path, 'r') as f:
            lines = f.readlines()
            self.NN_list = [
                    os.path.join(NN_path, i.split()[0]) for i in lines
                    ]
            self.NoneNN_list = [
                os.path.join(NoneNN_path, i.split()[0]) for i in lines
            ]
            self.label_list = [int(i.split()[1])-1 for i in lines]
        self.transform = transform
        self.loader = loader
    
    
    def __getitem__(self, index):
        NN_path = self.NN_list[index]
        NoneNN_path = self.NoneNN_list[index]
        label = self.label_list[index]
        NN = self.loader(NN_path)
        NoneNN = self.loader(NoneNN_path)
        img = average_(NN, NoneNN)
        if(self.transform is not None):
            img = self.transform(data)
        return img, label
    """
    def __init__(self, data_path, label_path, loader=read_mat_1D, transform=None):
        self.data = loader(data_path)
        self.label_list = np.loadtxt(label_path)
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index, :]
        label = self.label_list[index] - 1
        return data, label
    """
    

    def __len__(self):
        return len(self.label_list)
        # return 960



if(__name__ == '__main__'):
    data_path = 'train_1NN_4050_log.mat'
    label_path = '../data/train_label.txt'

    md = MyDataset(data_path, label_path)
    kwargs = {}
    data_loader = torch.utils.data.DataLoader(md, batch_size=64, shuffle=True, **kwargs)
    for data, label in data_loader:
        print(data.size())
