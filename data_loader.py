import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.nn.init import xavier_normal_ as xavier_normal_

class PoolSet(Dataset):
    def __init__(self, p_x):
        ## input: torch.tensor (NOT CUDA TENSOR)
        self.len = len(p_x)
        self.x = p_x    ##[N, p]
    
    def __getitem__(self, index):
        return self.x[index]
    
    def __len__(self):
        return self.len


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
      xavier_normal_(m.weight)
      m.bias.data.fill_(0.0)