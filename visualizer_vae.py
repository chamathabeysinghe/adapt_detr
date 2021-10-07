import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# import datasets
import util.misc as utils
from datasets import build_dataset
# from datasets import build_dataset, get_coco_api_from_dataset
from models_vae import build_model


class Args:
    dataset_file = 'ant2'
    data_path = '/dice1-data/home/cabe0006/cvpr_experiments/cvpr_data/detection_dataset_small'
    masks = False
    batch_size = 1
    num_workers = 2


args = Args()

device = torch.device('cuda')
model, criterion = build_model(None)
model.to(device)

checkpoint_path = '/dice1-data/home/cabe0006/cvpr_experiments/vae_output/basic_vae_on_ants/checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=True)

dataset_test = build_dataset(image_set='train', args=args)
sampler_test = torch.utils.data.RandomSampler(dataset_test)
batch_sampler_test = torch.utils.data.BatchSampler(
    sampler_test, args.batch_size, drop_last=True)
data_loader_test = DataLoader(dataset_test, batch_sampler=batch_sampler_test,
                               collate_fn=utils.collate_fn, num_workers=args.num_workers)


invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

model.eval()
with torch.no_grad():
    for idx, data in enumerate(iter(data_loader_test)):
        imgs = data[0].tensors
        # imgs, _ = data
        imgs = imgs.to(device)
        out, mu, logVAR = model(imgs)

        inv_img = invTrans(imgs[0])
        img = np.transpose(inv_img.cpu().numpy(), [1,2,0])
        plt.subplot(121)
        plt.imshow(np.squeeze(img))

        inv_out = invTrans(out[0])
        outimg = np.transpose(inv_out.cpu().numpy(), [1,2,0])
        plt.subplot(122)
        plt.imshow(np.squeeze(outimg))
        # plt.savefig('foo.png')
        plt.show()
        if idx > 10:
            break


