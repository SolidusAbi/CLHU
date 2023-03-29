import os
import torch
from torch import tensor
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np


class HSIDataset(Dataset):
    def __init__(self):
        super(HSIDataset, self).__init__()

    def preprocessing(self, X:np.ndarray, max_value=-1):
        '''
            Preprocessing the dataset for removing high-frequency noise. 
            This preprocessing consists of three steps:
                1. Median filter in the spatial domain.
                2. Moving average filter in the spectral domain. (No!)
                3. Normalization of the data.

            Parameters
            ----------
                X : np.ndarray, shape (nRow, nCol, nBand)
                    HSI Cube.
        '''
        from skimage.filters import median
        from utils import moving_average

        if max_value == -1:
            max_value = X.max() + 1e-3

        X = median(X, footprint=np.ones((3,3,1)))
        X = moving_average(X.reshape(-1, X.shape[-1]), 3, padding_size=2).reshape(X.shape[0], X.shape[1], -1)
        return X / max_value

# Jasper Ridge dataset
class JasperRidge(HSIDataset):
    def __init__(self, root_dir, transform=None):
        super(JasperRidge, self).__init__()
        data = sio.loadmat(os.path.join(root_dir, 'jasperRidge2_R198.mat'))
        y = sio.loadmat(os.path.join(root_dir, 'GroundTruth/end4.mat'))

        self.n_row, self.n_col = data['nRow'].item(), data['nCol'].item()

        self.X = data['Y'].T.reshape(self.n_row, self.n_col, -1) # (nRow, nCol, nBand)
        self.X = self.preprocessing(self.X).reshape(-1, self.X.shape[-1]) # (nRow*nCol, nBand)
        self.X = tensor(self.X, dtype=torch.float32)

        self.E = tensor(y['M'].T, dtype=torch.float32) # (nEndmember, nBand)
        self.A = tensor(y['A'].T, dtype=torch.float32) # (nRow*nCol, nEndmember)
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
        return self.A.reshape(self.n_row, self.n_col, -1)

    def image(self):
        return self.X.reshape(self.n_row, self.n_col, -1)
    
# Samson dataset
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

        data = sio.loadmat(os.path.join(root_dir, 'Data_Matlab/samson_1.mat'))
        y = sio.loadmat(os.path.join(root_dir, 'GroundTruth/end3.mat'))


        self.n_row, self.n_col , self.n_bands = data['nRow'].item(), data['nCol'].item(), data['nBand'].item()

        self.X = data['V'].T.reshape(self.n_row, self.n_col, -1) # (nRow, nCol, nBand)
        self.X = self.preprocessing(self.X).reshape(-1, self.X.shape[-1]) # (nRow*nCol, nBand)
        self.X = tensor(self.X, dtype=torch.float32)

        self.E = tensor(y['M'].T, dtype=torch.float32) # (nEndmember, nBand)
        self.A = tensor(y['A'].T, dtype=torch.float32) # (nRow*nCol, nEndmember)
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
        return self.A.reshape(self.n_row, self.n_col, -1)

    def image(self):
        return self.X.reshape(self.n_row, self.n_col, -1)
    
# Urban dataset
class Urban(HSIDataset):
    def __init__(self, root_dir, transform=None):
        super(Urban, self).__init__()

        data = sio.loadmat(os.path.join(root_dir, 'Urban_R162.mat'))
        y = sio.loadmat(os.path.join(root_dir, 'groundTruth/end4_groundTruth.mat'))
        # y = sio.loadmat(os.path.join(root_dir, 'groundTruth_Urban_end5/end5_groundTruth.mat'))


        self.n_row, self.n_col, self.n_bands = data['nRow'].item(), data['nCol'].item(), len(data['SlectBands'])

        self.X = data['Y'].T.reshape(self.n_row, self.n_col, -1) # (nRow, nCol, nBand)
        self.X = self.preprocessing(self.X).reshape(-1, self.X.shape[-1]) # (nRow*nCol, nBand)
        self.X = tensor(self.X, dtype=torch.float32)

        self.E = tensor(y['M'].T, dtype=torch.float32) # (nEndmember, nBand)
        self.A = tensor(y['A'].T, dtype=torch.float32) # (nRow*nCol, nEndmember)
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
        return self.A.reshape(self.n_row, self.n_col, -1)

    def image(self):
        return self.X.reshape(self.n_row, self.n_col, -1)
    
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
        return self.X.reshape(self.n_row, self.n_col, -1)
    
class DLR_HySU(HSIDataset):
    '''
        DLR_HySU Dataset.

        Cerra, D., Pato, M., Alonso, K., Köhler, C., Schneider, M., de los Reyes, R., ... & Müller, R. (2021). 
        Dlr hysu—a benchmark dataset for spectral unmixing. Remote Sensing, 13(13), 2559.
        
        Attributes:
        ----------
            n_row: int
                Number of rows.
            n_col: int
                Number of columns.
            n_bands: int
                Number of bands.
            n_endmembers: int
                Number of endmembers.
            wv: ndarray, shape (n_bands)
                Wavelength in μm (1e-6).
            X: Tensor, shape (n_row*n_col, n_bands)
                hyperspectral Cube.
            E: Tensor, shape (n_endmembers, n_bands)
                Endmembers of the different material, Grass included.
            A: Tensor, shape (n_row*n_col, n_endmembers-1) 
                Abundance maps without the grass.

    '''
    def __init__(self, root_dir, transform=None):
        super(DLR_HySU, self).__init__()

        data = sio.loadmat(os.path.join(root_dir, 'dlr_hysu_R135.mat'))
        y = sio.loadmat(os.path.join(root_dir, 'gt.mat'))

        self.n_row, self.n_col , self.n_bands = data['nRow'].item(), data['nCol'].item(), data['nBand'].item()
        self.wv = data['wavelength'].T.reshape(-1)

        self.X = data['Y'].T.reshape(self.n_row, self.n_col, -1) # (nRow, nCol, nBand)
        self.X = self.preprocessing(self.X).reshape(-1, self.X.shape[-1]) # (nRow*nCol, nBand)
        self.X = tensor(self.X, dtype=torch.float32)

        self.E = tensor(y['M'].T, dtype=torch.float32) # (nEndmember, nBand)
        self.E = self.E / (self.E.max() + 1e-3)
        self.A = tensor(y['A'].T, dtype=torch.float32) # (nRow*nCol, nEndmember)
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
        return self.A.reshape(self.n_row, self.n_col, -1)

    def image(self):
        return self.X.reshape(self.n_row, self.n_col, -1)