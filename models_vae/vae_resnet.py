# vgg = models.resnet50()
# summary(vgg, (3, 542, 1042))

"""
The following is an import of PyTorch libraries.
"""
import torch
from torch import nn
from models_vae.encoder import build_encoder
from models_vae.decoder import build_decoder
import torch.nn.functional as F
"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
A Convolutional Variational Autoencoder
"""


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cnv_mu = nn.Conv2d(2048, 16, 3, padding=1)
        self.cnv_var = nn.Conv2d(2048, 16, 3, padding=1)

    def encode(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x_encoded = F.relu(self.encoder(x))
        mu, log_var = F.tanh(self.cnv_mu(x_encoded)), F.tanh(self.cnv_var(x_encoded))
        return mu, log_var

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = self.decoder(z)
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        out = self.decode(z)
        return out, mu, logVar


def build(args):
    encoder = build_encoder(args)
    decoder = build_decoder()
    model = VAE(encoder, decoder)
    criterion = nn.BCELoss(size_average=False)
    return model, criterion