# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Ant dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/Ant_utils.py
"""
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

import datasets.transforms as T


class AntDetection(Dataset):
    def __init__(self, root, img_folders, image_set):
        self._train_transforms_torch = get_train_transforms_torch()
        self._val_transforms_torch = get_val_transforms_torch()
        self.image_set = image_set
        self.image_paths = []
        for dir in img_folders:
            files = [os.path.join(dir, f) for f in filter(
                lambda x: 'jpg' in x or 'png' in x or 'jpeg' in x, os.listdir(os.path.join(root, dir)))]
            self.image_paths += files
        self.root = root

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        img, _ = self._train_transforms_torch(img, {})
        return img


def get_train_transforms_torch():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize([(1024, 542)], max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize([(1024, 542)], max_size=1333),
            ])
        ),
        normalize,
    ])


def get_val_transforms_torch():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.RandomResize([(1024, 542)], max_size=1333),
        normalize,
    ])


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided Ant path {root} does not exist'
    PATHS = {
        "train": ['train', 'val'],
        "test": ['val'],
    }

    img_folders = PATHS[image_set]
    dataset = AntDetection(root, img_folders, image_set)
    return dataset
