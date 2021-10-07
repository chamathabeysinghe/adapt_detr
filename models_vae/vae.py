# vgg = models.resnet50()
# summary(vgg, (3, 542, 1042))

"""
The following is an import of PyTorch libraries.
"""
import torch
import torch.nn.functional as F
from torch import nn

"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
A Convolutional Variational Autoencoder
"""


class VAE(nn.Module):
    def __init__(self, imgChannels=3):
        super(VAE, self).__init__()
        self.down1 = nn.Conv2d(imgChannels, 16, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(16, 16, 3, stride=2, padding=1)

        self.same1 = nn.Conv2d(16, 16, 3, padding=1)
        self.same2 = nn.Conv2d(16, 16, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(16, imgChannels, 3, stride=2, padding=1)

    def encoder(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x = F.relu(self.down3(x))
        mu = self.same1(x)
        logVar = self.same2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.up1(z, output_size=(136, 256)))
        x = F.relu(self.up2(x, output_size=(271, 512)))
        x = torch.sigmoid(self.up3(x, output_size=(542, 1024)))
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

def build(args):
    model = VAE()
    criterion = nn.BCELoss(size_average=False)
    return model, criterion