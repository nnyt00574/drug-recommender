import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim,128), nn.ReLU(), nn.Linear(128,32))
        self.dec = nn.Sequential(nn.Linear(32,128), nn.ReLU(), nn.Linear(128,dim))

    def forward(self,x):
        z = self.enc(x)
        return self.dec(z), z