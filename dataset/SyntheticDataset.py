from scipy import io as sio
from .HSIDataset import HSIDataset
import numpy as np
import torch, os

class SyntheticDataset(HSIDataset):
    def __init__(self, root_dir, transform=None):
        super(SyntheticDataset, self).__init__()
        data = sio.loadmat(os.path.join(root_dir, "snr20/Y.mat"))

        self.n_row, self.n_col , self.n_bands = data['nRow'].item(), data['nCol'].item(), data['nBand'].item()
        self.X = np.abs(data['Y'].T) # Because of the noise, there are negative values
        self.X = self.X.reshape(self.n_row, self.n_col, -1)
        self.X = self.preprocessing(self.X, max_value=1).reshape(-1, self.X.shape[-1]) # (nRow*nCol, nBand)

        self.E = sio.loadmat(os.path.join(root_dir, "snr20/M.mat"))['M_avg'].T
        self.A = sio.loadmat(os.path.join(root_dir, "snr20/A.mat"))['A'].T

        self.X = torch.tensor(self.X, dtype=torch.float32)
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

    def image(self):
        return self.X.numpy().reshape(self.n_row, self.n_col, -1, order='F')