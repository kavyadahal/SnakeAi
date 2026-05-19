import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import torch.nn as nn
import numpy as np

class Linear_Qnet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size,hidden_size)
        self.linear1 = torch.nn.Linear(hidden_size,output_size)    