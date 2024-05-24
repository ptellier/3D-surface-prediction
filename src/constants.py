import torch.cuda

TORCH_DEVICE = 'cpu'
if torch.backends.mps.is_available():
    TORCH_DEVICE = 'mps'
if torch.cuda.is_available():
    TORCH_DEVICE = 'cuda'

# Datasets
MANUAL_DATASET_FOLDER_PATH = './datasets/manual_dataset'
