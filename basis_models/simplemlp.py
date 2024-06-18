
import torch.nn as nn
import torch 
import torch.optim as optim
class SimpleMLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(3,32),
        nn.LeakyReLU(),
        nn.Dropout(0.3),
        nn.Linear(32,16),
        nn.Tanh())
    self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-3)

  def forward(self, dir_pp_normalized):
    out = self.model(dir_pp_normalized)
    return out

  def capture(self):
    return {
      'model_state_dict': self.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict()
    }
  
  def initialize(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                m.bias.data.fill_(0.0)