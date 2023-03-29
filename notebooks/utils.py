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
def plot_endmembers(E: np.ndarray, wv:np.ndarray = None, labels:list = None, figsize:tuple = (7,5), ticks_range:tuple=(0, 1), n_ticks:int=5):
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
            ticks_range : tuple, optional
                Range of yticks. Default is (0, 1).
            n_ticks : int, optional
                Number of yticks. Default is 5.
    '''
    ticks_formatter = plt.FormatStrFormatter('%.2f')

    n_endmembers, n_bands = E.shape
    if labels is None:
        labels = list(map(lambda x: r'$E_{{{}}}$'.format(x), range(1, n_endmembers+1)))

    with plt.style.context(("seaborn-colorblind")):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ticks = np.linspace(*ticks_range, n_ticks)
        if wv is None:
            ax.plot(E.T, label=labels)
            ax.set_xlabel('Bands', fontsize='x-large')
        else:
            ax.plot(wv, E.T, label=labels)
            ax.set_xlabel('Wavelength (nm)', fontsize='x-large')

        ax.set_ylabel('Reflectance', fontsize='x-large')           
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(ticks_formatter) # set format in y ticks labels
        ax.set_ylim(ticks_range[0] - 0.025, ticks_range[1] + 0.025)
        ax.set_xlim(0 - 1.5, n_bands + 1.5)
        ax.tick_params(axis='both', labelsize='large')
    
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6, fontsize='large', borderpad=-.25)
        fig.tight_layout(pad=(((n_endmembers-1)//6)+1)*2) # padding based on the endmembers number

    return fig

def show_abundance(A, labels:list = None, figsize:tuple=(7,5)):
    '''
        Show abundance maps.

        Parameters
        ----------
            A : 3-D array, shape (n_rows, n_cols, n_endmembers)
                Abundance maps.
            labels : list, optional
                Labels for endmembers. Default is None.
            figsize : tuple, optional
                Figure size. Default is (7,5).
    '''
    _, _, n_endmembers = A.shape

    if labels is None:
        labels = list(map(lambda x: r'$E_{{{}}}$'.format(x), range(1, n_endmembers+1)))
        
    ticks_formatter = plt.FormatStrFormatter('%.1f')
    fig = plt.figure(figsize=(7,5))
    for i in range(n_endmembers):
        data = A[:,:,i].T
        plt.subplot(3,4,i+1)
        plt.imshow(data, cmap='viridis')
        plt.axis('off')
        plt.title(labels[i], fontsize='x-large')
        cb = plt.colorbar(format=ticks_formatter, ticks=[data.min() + 1e-3, data.max() - 1e-3],
                         orientation='horizontal', fraction=0.1, pad=0.01)

    plt.tight_layout()
    return fig