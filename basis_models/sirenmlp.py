import torch.nn as nn
import numpy as np
import torch 
from utils.general_utils import get_expon_lr_func

def sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

class SINLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, freq: float):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.freq = freq

    def forward(self, x):
        return torch.sin(self.freq * self.layer(x))

class SirenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            SINLayer(3, 64, 60.),
            SINLayer(64, 64, 10.),
            SINLayer(64, 64, 1.),
            nn.Linear(64, 16), # DIMREC
            nn.Tanh())
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-3)

        self.lr_scheduler = get_expon_lr_func(lr_init=2e-3,
                                            lr_final=5e-4,
                                            lr_delay_mult=0.01,
                                            max_steps=30_000)

    def forward(self, dir_pp_normalized):
        out = self.model(dir_pp_normalized)
        return out
    
    def capture(self):
        return {
        'model_state_dict': self.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict()
        }

    def initialize(self):
        for i in range(4):
            self.model[i].apply(sine_init)
        self.model[0].apply(first_layer_sine_init)
        self.model[3].apply(frequency_init(30))
    
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            lr = self.lr_scheduler(iteration)
            param_group['lr'] = lr
        return lr