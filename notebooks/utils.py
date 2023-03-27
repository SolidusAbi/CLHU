import numpy as np

def moving_average(a:np.ndarray, n=3, padding_size=2):
    ''' 
        Moving average filter for 1-D array.

        Parameters
        ----------
        a : 1-D array, shape (batch_size, n_features)
            Input array.
        n : int, optional
            Window size. Default is 3.
        padding_size : int, optional
            Padding size. Default is 2.
    
    '''
    if padding_size > 0:
        a = np.pad(a, ((0,0), (padding_size//2, padding_size//2)), 'edge')

    ret = np.cumsum(a, axis=1, dtype=float)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:] / n

from matplotlib import pyplot as plt
def plot_endmembers(E: np.ndarray, wv:np.ndarray = None, labels:list = None, figsize:tuple = (7,5)):
    '''
        Plot endmembers.

        Parameters
        ----------
            E : 2-D array, shape (n_endmembers, n_bands)
                Endmembers.
            wv : 1-D array, optional, shape (n_bands)
                Wavelengths in nm. Default is None.
            labels : list, optional
                Labels for endmembers. Default is None.
            figsize : tuple, optional
                Figure size. Default is (7,5).

    '''

    n_endmembers, _ = E.shape
    if labels is None:
        labels = list(map(lambda x: f'$E_{x}$', range(1, n_endmembers+1)))

    with plt.style.context(("seaborn-colorblind")):
        fig = plt.figure(figsize=(7, 5))
        if wv is None:
            plt.plot(E.T, label=labels)
            plt.xlabel('Bands', fontsize='x-large')
        else:
            plt.plot(wv, E.T, label=labels)
            plt.xlabel('Wavelength (nm)', fontsize='x-large')
        plt.ylabel('Reflectance', fontsize='x-large')           
        plt.legend(fontsize='large', loc='upper left')
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')

        plt.tight_layout()

    return fig