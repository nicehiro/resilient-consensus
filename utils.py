import torch
from torch.utils.tensorboard import SummaryWriter


if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
    
    
device = torch.device('cpu')

# summary writer for tensorboard
writer = SummaryWriter('logs')