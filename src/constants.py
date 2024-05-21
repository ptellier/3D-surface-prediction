import torch.cuda

TORCH_DEVICE = 'cpu'
if torch.backends.mps.is_available():
    TORCH_DEVICE = 'mps'
if torch.cuda.is_available():
    TORCH_DEVICE = 'cuda'
