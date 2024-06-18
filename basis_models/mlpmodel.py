import json
import tinycudann as tcnn
import torch.nn as nn
import torch as pt
import torch.optim as optim
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init

class Attention(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.linear = nn.Linear(input_dim, output_dim)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(pt.float32)
    attention = self.softmax(self.linear(x))
    return attention * x


class MLPModel(nn.Module):
  def __init__(self):
    super().__init__()
    with open("basis_models/config_hash.json") as config_file:
      config = json.load(config_file)
    # DIMREC
    self.basis_tinycudann = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=16, encoding_config=config["encoding_tinycudann"], network_config=config["network_tinycudann"]).to(pt.device("cuda"))

    self.optimizer = pt.optim.AdamW(self.parameters(), lr=1e-3)
    
    #print('Basis Network:',self.model)
  def forward(self, dir_pp_normalized):
    
    out_basis = self.basis_tinycudann(dir_pp_normalized)

    return out_basis

  def initialize(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
          m.bias.data.fill_(0.0)

  def capture(self):
    return {
      'model_state_dict': self.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict()
    }
