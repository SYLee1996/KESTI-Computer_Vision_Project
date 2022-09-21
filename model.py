import torch 
import torch.nn as nn

class Deep_SVDD(nn.Module):
    def __init__(self, in_size):
        super(Deep_SVDD, self).__init__()
        self.conv1 = nn.Linear(in_size, int(in_size/2))
        self.conv2 = nn.Linear(int(in_size/2), int(in_size/4))
        self.conv3 = nn.Linear(int(in_size/4), int(in_size/8))
        
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        z = self.relu(x)
        return z

class C_AutoEncoder(nn.Module):
    def __init__(self, in_size):
        super(C_AutoEncoder, self).__init__()
        
        self.conv1 = nn.Linear(in_size, int(in_size/2))
        self.conv2 = nn.Linear(int(in_size/2), int(in_size/4))
        self.conv3 = nn.Linear(int(in_size/4), int(in_size/8))
        
        self.deconv1 = nn.Linear(int(in_size/8), int(in_size/4))
        self.deconv2 = nn.Linear(int(in_size/4), int(in_size/2))
        self.deconv3 = nn.Linear(int(in_size/2), in_size)
        
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def encoder(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        z = self.relu(x)
        return z
    
    def decoder(self, z):
        z = self.deconv1(z)
        z = self.relu(z)
        z = self.deconv2(z)
        z = self.relu(z)
        z = self.deconv3(z)
        x_hat = self.sigmoid(self.relu(z))
        return x_hat
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

