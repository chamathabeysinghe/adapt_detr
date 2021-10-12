import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# import datasets
from datasets import build_dataset
# from datasets import build_dataset, get_coco_api_from_dataset
from models_vae import build_model
import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class Args:
    dataset_file = 'vae_ant'
    data_path = '/dice1-data/home/cabe0006/cvpr_experiments/cvpr_data/detection_dataset_small'
    masks = False
    batch_size = 1
    num_workers = 2
    backbone='resnet50'
    dilation=False


args = Args()

device = torch.device('cpu')
model, criterion = build_model(args)
model.to(device)

checkpoint_path = '/Users/cabe0006/Projects/monash/cvpr_experiments/vae_resnet_cifar/checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=True)

trainset = torchvision.datasets.CIFAR10(root='/Users/cabe0006/Projects/monash/data/cifar', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)


invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

model.eval()
with torch.no_grad():
    for idx, data in enumerate(iter(trainloader)):
        print(idx)
        # imgs = data[0].tensors
        # imgs, _ = data
        imgs = data[0].to(device)
        out, mu, logVAR = model(imgs)
        inv_img = imgs[0]/2+0.5
        img = np.transpose(inv_img.cpu().numpy(), [1,2,0])
        plt.subplot(121)
        show_img1 = np.squeeze(img)
        plt.imshow(show_img1)

        inv_out = out[0]/2.0 + 0.5
        outimg = np.transpose(inv_out.cpu().numpy(), [1,2,0])
        plt.subplot(122)
        show_img2 = np.squeeze(outimg)
        plt.imshow(show_img2)

        vis = (np.concatenate((show_img1, show_img2), axis=1) * 255).astype(np.uint8)
        # plt.imshow(vis)
        cv2.imwrite(f'/Users/cabe0006/Projects/monash/cvpr_experiments/vae_resnet_cifar/predictions/{idx}.png', vis)

        # plt.savefig('foo.png')
        plt.show()
        if idx > 500:
            break


