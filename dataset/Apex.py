from .HSIDataset import HSIDataset
import os
import scipy.io as sio
from torch import tensor
import torch
import numpy as np

# Jasper Ridge dataset
class Apex(HSIDataset):
    def __init__(self, root_dir, transform=None):
        super(Apex, self).__init__()
        data = sio.loadmat(os.path.join(root_dir, 'apex_dataset.mat'))
        selectBands = sio.loadmat(os.path.join(root_dir, 'slectBands.mat'))['SlectBands'].squeeze().astype(bool)

        self.n_row, self.n_col = (110, 110)

        self.X = tensor(data['Y'].T, dtype=torch.float32) # (nRow*nCol, nBand)
        self.E = tensor(data['M'].T, dtype=torch.float32) # (nEndmember, nBand)
        self.A = tensor(data['A'].T, dtype=torch.float32) # (nRow*nCol, nEndmember)
        self.wv = np.linspace(413, 2420, self.X.shape[-1], dtype=int)

        # self.X = self.X[:, selectBands.squeeze()]
        # self.E = self.E[:, selectBands.squeeze()]
        # self.wv = self.wv[selectBands.squeeze()]
        
        self.n_bands = self.X.shape[1]
        self.n_endmembers = self.E.shape[0]
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
        return self.A.numpy().reshape(self.n_row, self.n_col, -1)

    def image(self):
        return self.X.numpy().reshape(self.n_row, self.n_col, -1)