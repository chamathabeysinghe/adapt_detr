import torch
from typing import Iterable
from tqdm import tqdm
import torch.nn.functional as F
import util.misc as utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int
                    ):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        imgs = samples.tensors
        imgs = imgs.to(device)

        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = model(imgs)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        bce_loss = F.mse_loss(out, imgs, size_average=False)
        loss = bce_loss + kl_divergence
        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        vae_loss_dict = {
            'bce_loss': bce_loss,
            'kl_loss': kl_divergence,
            'vae_loss': loss
        }
        vae_loss_dict_reduced = utils.reduce_dict(vae_loss_dict)
        metric_logger.update(**vae_loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


    # with tqdm(data_loader, unit='batch') as tepoch:
    #     for data in tepoch:
    #         tepoch.set_description(f"Epoch {epoch}")
    #         imgs = data[0].tensors
    #         # imgs, _ = data
    #         imgs = imgs.to(device)
    #
    #         # Feeding a batch of images into the network to obtain the output image, mu, and logVar
    #         out, mu, logVar = model(imgs)
    #
    #         # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
    #         kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
    #         bce_loss = F.binary_cross_entropy(out, imgs, size_average=False)
    #         loss = bce_loss + kl_divergence
    #         # Backpropagation based on the loss
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         vae_loss_dict = {
    #             'bce_loss': bce_loss,
    #             'kl_loss': kl_divergence,
    #             'vae_loss': loss
    #         }
    #         vae_loss_dict_reduced = utils.reduce_dict(vae_loss_dict)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




