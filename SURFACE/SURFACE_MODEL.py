import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_size, int(in_size/2))
        self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
        self.linear3 = nn.Linear(int(in_size/4), latent_size)
        self.relu = nn.ReLU(True)
        
    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size/4))
        self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
        self.linear3 = nn.Linear(int(out_size/2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w
    
class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.w_size = config['w_size']
        self.z_size = config['z_size']
        self.encoder = Encoder(self.w_size, self.z_size)
        self.decoder1 = Decoder(self.z_size, self.w_size)
        self.decoder2 = Decoder(self.z_size, self.w_size)
    
    def forward(self, batch):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        
        return w1, w2, w3
        