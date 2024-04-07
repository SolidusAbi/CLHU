from scipy import io as sio
from .HSIDataset import HSIDataset
import numpy as np
import torch, os
from torch.nn.functional import hardtanh 

from enum import Enum

class SyntheticDatasetType(Enum):
    SNR10 = 1,
    SNR20 = 2,
    SNR30 = 3,
    SNR40 = 4

class SyntheticDataset(HSIDataset):
    def __init__(self, root_dir, type:SyntheticDatasetType=SyntheticDatasetType.SNR30, transform=None):
        super(SyntheticDataset, self).__init__()
    
        if type == SyntheticDatasetType.SNR10:
            snr_path = "snr10"
        elif type == SyntheticDatasetType.SNR20:
            snr_path = "snr20"
        elif type == SyntheticDatasetType.SNR30:
            snr_path = "snr30"
        elif type == SyntheticDatasetType.SNR40:
            snr_path = "snr40"
        else:
            raise ValueError("Invalid type")

        data = sio.loadmat(os.path.join(root_dir, "{}/Y.mat".format(snr_path)))
        self.n_row, self.n_col , self.n_bands = data['nRow'].item(), data['nCol'].item(), data['nBand'].item()
        self.X = data['Y'].T
    
        self.E = sio.loadmat(os.path.join(root_dir, "{}/M.mat".format(snr_path)))['M_avg'].T
        self.A = sio.loadmat(os.path.join(root_dir, "{}/A.mat".format(snr_path)))['A'].T

        self.X = hardtanh(torch.tensor(self.X, dtype=torch.float32), 0, .99) # Because of the noise, there are negative values
        self.E = torch.tensor(self.E, dtype=torch.float32)
        self.A = torch.tensor(self.A, dtype=torch.float32)
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

    def image(self, order='F'):
        return self.X.numpy().reshape(self.n_row, self.n_col, -1, order=order)