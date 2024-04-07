import os, sys
from matplotlib import pyplot as plt

plt.style.use('seaborn-v0_8')
project_root_dir = os.path.dirname(__file__)

# Dataset path
# DATASET_PATH = '/media/abian/Extreme SSD/WorkSpace/Dataset/HSI/'
DATASET_PATH = '/home/abian/Data/Dataset/HSI/'
JasperRidge_PATH = os.path.join(DATASET_PATH, 'JasperRidge') 
Samson_PATH = os.path.join(DATASET_PATH, 'Samson')
Urban_PATH = os.path.join(DATASET_PATH, 'Urban')
Cuprite_PATH = os.path.join(DATASET_PATH, 'Cuprite')
DLR_HySU_PATH = os.path.join(DATASET_PATH, 'DLR_HySU')
Synthetic_PATH = os.path.join(DATASET_PATH, 'SyntheticData')
Apex_PATH = os.path.join(DATASET_PATH, 'Apex/')
NerveFat_PATH = '/home/abian/Data/Dataset/HSI/Japan/Unmixing/Compovision/HSI/A/data'

# Results path
RESULTS_PATH = '/media/abian/Extreme SSD/Thesis/CLHU/review/'
# RESULTS_PATH = os.path.join(project_root_dir, 'data/results/')
IMG_PATH = os.path.join(project_root_dir, 'data/img/')

# Dependencies
hyspeclab_dir = os.path.join(project_root_dir, 'modules/HySpecLab')
if hyspeclab_dir not in sys.path:
    sys.path.append(hyspeclab_dir)

ipdl_dir = os.path.join(project_root_dir, 'modules/IPDL')
if ipdl_dir not in sys.path:
    sys.path.append(ipdl_dir)

hsi2rgb_dir = os.path.join(project_root_dir, 'modules/HSI2RGB')
if hsi2rgb_dir not in sys.path:
    sys.path.append(hsi2rgb_dir)
    
