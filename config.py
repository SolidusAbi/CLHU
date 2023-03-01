import os, sys
project_root_dir = os.path.dirname(__file__)

# Dataset path
DATASET_PATH = '/home/abian/Data/Dataset/HSI/'
JasperRidge_PATH = os.path.join(DATASET_PATH, 'JasperRidge') 
Samson_PATH = os.path.join(DATASET_PATH, 'Samson')
Urban_PATH = os.path.join(DATASET_PATH, 'Urban')

# Results path
RESULTS_PATH = os.path.join(project_root_dir, 'data/results/')
IMG_PATH = os.path.join(project_root_dir, 'data/img/')

# Dependencies
hyspeclab_dir = os.path.join(project_root_dir, 'modules/HySpecLab')
if hyspeclab_dir not in sys.path:
    sys.path.append(hyspeclab_dir)

ipdl_dir = os.path.join(project_root_dir, 'modules/IPDL')
if ipdl_dir not in sys.path:
    sys.path.append(ipdl_dir)