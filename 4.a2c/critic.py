import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        # input - 2
        # output - 3 (action probabilities)
        self.fc1 = nn.Linear(8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        #self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128,1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        return x