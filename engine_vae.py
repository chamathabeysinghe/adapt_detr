import torch
from typing import Iterable
from tqdm import tqdm
import torch.nn.functional as F


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int
                    ):
    model.train()
    criterion.train()
    with tqdm(data_loader, unit='batch') as tepoch:
        for data in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            imgs = data[0].tensors
            # imgs, _ = data
            imgs = imgs.to(device)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logVar = model(imgs)

            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print('Epoch {}: Loss {}'.format(epoch, loss))

    # TODO handle logging properly using utils
    return {'vae_loss': loss.item()}




