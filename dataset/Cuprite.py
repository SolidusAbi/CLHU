from .HSIDataset import HSIDataset
import os
import scipy.io as sio
from torch import tensor
import torch

class Cuprite(HSIDataset):
    ''' 
        Cuprite dataset. The dataset has been preprocessed by removing the first two slides because
        they are not useful.
    '''
    def __init__(self, root_dir, transform=None):
        super(Cuprite, self).__init__()

        data = sio.loadmat(os.path.join(root_dir, 'CupriteS1_R188.mat'))
        y = sio.loadmat(os.path.join(root_dir, 'groundTruth_Cuprite_end12/groundTruth_Cuprite_nEnd12.mat'))

        self.n_row, self.n_col, self.n_bands = data['nRow'].item(), data['nCol'].item(), len(data['SlectBands'])

        self.X = data['Y'].T.reshape(self.n_row, self.n_col, -1, order='F') # (nRow, nCol, nBand)
        self.X = self.X[:,:, 2:] # Remove the first two slides
        self.n_bands = self.n_bands - 2
        self.X = self.preprocessing(self.X).reshape(-1, self.X.shape[-1]) # (nRow*nCol, nBand)
        self.X = tensor(self.X, dtype=torch.float32)

        self.E = tensor(y['M'].T, dtype=torch.float32) # (nEndmember, nBand)
        self.n_endmembers = self.E.shape[0]

        select = data['SlectBands'].reshape(-1).astype(int) - 1 # Indexing in matlab starts from 1
        self.E = self.E[:, select[2:]]        

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
        return None

    def image(self):
        return self.X.numpy().reshape(self.n_row, self.n_col, -1, order='F')