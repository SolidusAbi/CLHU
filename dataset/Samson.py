from .HSIDataset import HSIDataset
import os
import scipy.io as sio
from torch import tensor
import torch
import numpy as np

class Samson(HSIDataset):
    ''' 
        Samson dataset

        Attributes
        ----------
            n_row : int
                Number of rows.
            n_col : int
                Number of columns.
            n_bands : int
                Number of bands.
            n_endmembers : int
                Number of endmembers.
            X : torch.Tensor, shape (nRow*nCol, nBand)
                HSI Cube.
            E : torch.Tensor, shape (nEndmember, nBand)
                Endmembers.
            A : torch.Tensor, shape (nRow*nCol, nEndmember)
                Abundance.
    '''
    def __init__(self, root_dir, transform=None):
        super(Samson, self).__init__()

        # data = sio.loadmat(os.path.join(root_dir, 'Data_Matlab/samson_1.mat'))
        # y = sio.loadmat(os.path.join(root_dir, 'GroundTruth/end3.mat'))
        data = sio.loadmat(os.path.join(root_dir, 'samson_dataset.mat'))

        # self.n_row, self.n_col , self.n_bands = data['nRow'].item(), data['nCol'].item(), data['nBand'].item()
        self.n_row, self.n_col , self.n_bands = 95,95,156

        # self.X = data['Y'].T.reshape(self.n_row, self.n_col, -1) # (nRow, nCol, nBand)
        # self.X = self.preprocessing(self.X).reshape(-1, self.X.shape[-1]) # (nRow*nCol, nBand)
        # self.X = tensor(self.X, dtype=torch.float32)

        self.X = tensor(data['Y'], dtype=torch.float32).T
        self.E = tensor(data['M'], dtype=torch.float32).T
        self.A = tensor(data['A'], dtype=torch.float32).T
        # self.E = tensor(y['M'].T, dtype=torch.float32) # (nEndmember, nBand)
        # self.A = y['A'].T.reshape(self.n_row, self.n_col, -1, order='C') # (nRow*nCol, nEndmember)
        # self.A = torch.tensor(y['A'].reshape(self.n_row*self.n_col, -1, order='')) # (nRow*nCol, nEndmember)
        # # self.A = tensor(y['A'].T, dtype=torch.float32) # (nRow*nCol, nEndmember)
        self.n_endmembers = self.E.shape[0]
        self.wv = np.linspace(401, 889, 156, dtype=int)
        self.transform = transform

    def __len__(self):
        return self.n_row * self.n_col

    def __getitem__(self, idx):
        sample = self.X[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def endmembers(self):
        return self.E

    def abundance(self):
        return self.A.reshape(self.n_row, self.n_col, -1)

    def image(self):
        return self.X.numpy().reshape(self.n_row, self.n_col, -1)