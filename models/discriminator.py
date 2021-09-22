from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=2048):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 256, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 8, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 8, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveMaxPool2d((16, 32)),
            nn.Flatten(),
            nn.Linear(8 * 16 * 32, 1024),
            nn.Linear(1024, 256),
            nn.Linear(256, 64),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def build_discriminator(args):
    in_channels = 2048
    discriminator = Discriminator(in_channels)
    criterion = nn.BCELoss()
    return discriminator, criterion
