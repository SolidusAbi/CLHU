from .HSIDataset import HSIDataset
import os
import scipy.io as sio
from torch import tensor
import torch
import pandas as pd
import numpy as np

class NerveFat(HSIDataset):
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
        super(NerveFat, self).__init__()

        data = np.load(os.path.join(root_dir, 'X_5.npy')) / 2**16# (nBand, nRow, nCol)
        self.wv = pd.read_csv(os.path.join(root_dir, 'wavelength.csv')).to_numpy().flatten().astype(int)

        # clipping data
        self.wv = self.wv[11:-39]
        data = data[11:-39]

        self.n_bands, self.n_row, self.n_col = data.shape
        self.X = torch.tensor(data).float().flatten(1).T # (nRow*nCol, nBand)
        self.X = self.X

        self.n_endmembers = -1
        self.E = None
        self.A = None
    
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
        return self.A

    def image(self):
        return self.X.numpy().reshape(self.n_row, self.n_col, -1)