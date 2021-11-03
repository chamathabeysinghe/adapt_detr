# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Ant dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/Ant_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as Ant_mask

import datasets.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from PIL import Image


class AntDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, return_masks, image_set):
        super(AntDetection, self).__init__(img_folder, ann_file)

        self._train_transforms_albm = get_train_transforms_albm()
        self._train_transforms_reduced = get_train_transforms_torch_reduced()
        self._train_transforms_torch = get_train_transforms_torch()
        self._val_transforms_torch = get_val_transforms_torch()
        self._test_minified_transforms_torch = get_test_minified_transforms_torch()

        self.prepare = ConvertAntPolysToMask(return_masks)
        self.image_set = image_set

    def __getitem__(self, idx):
        if self.image_set == 'train':
            img, target = super(AntDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.prepare(img, target)
            sample = {
                'image': np.asarray(img),
                'bboxes': target['boxes'],
                'labels': target['labels'],
                'orig_size': target['orig_size'],
                'size': target['size'],
                'image_id': target['image_id']
            }
            if self._train_transforms_albm is not None:
                sample = self._train_transforms_albm(**sample)
                img_augmented = Image.fromarray(sample['image'].astype('uint8'), mode='RGB')
                target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)
                target['labels'] = torch.tensor([0 for _ in range(len(sample['bboxes']))], dtype=torch.int64) #torch.tensor(sample['labels'], dtype=torch.float32)
                target['size'] = torch.tensor(img_augmented.size)
                if len(sample['bboxes']) > 0:
                    # print('BBoxes are greater than 0')
                    # print(target['size'])
                    img, target = self._train_transforms_reduced(img_augmented, target)
                else:
                    img, target = super(AntDetection, self).__getitem__(idx)
                    image_id = self.ids[idx]
                    target = {'image_id': image_id, 'annotations': target}
                    img, target = self.prepare(img, target)
                    img, target = self._train_transforms_torch(img, target)
            return img, target

        elif self.image_set == 'val' or self.image_set == 'test' or self.image_set == 'train_eval':
            img, target = super(AntDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.prepare(img, target)
            if self._val_transforms_torch is not None:
                img, target = self._val_transforms_torch(img, target)
            return img, target
        elif self.image_set == 'test_minified':
            img, target = super(AntDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.prepare(img, target)
            if self._test_minified_transforms_torch is not None:
                img, target = self._test_minified_transforms_torch(img, target)
            return img, target


def convert_Ant_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = Ant_mask.frPyObjects(polygons, height, width)
        mask = Ant_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertAntPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_Ant_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to Ant api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def get_train_transforms_albm():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.9),
            ], p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=1024, p=1.0),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            # ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_train_transforms_torch_reduced():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ])

    return T.Compose([
        T.RandomResize([(1024, 512)], max_size=1333),
        normalize,
    ])


def get_train_transforms_torch():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ])

    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize([(1024, 512)], max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize([(1024, 512)], max_size=1333),
            ])
        ),
        normalize,
    ])


def get_val_transforms_torch():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ])
    return T.Compose([
        T.RandomResize([(1024, 512)], max_size=1333),
        normalize,
    ])

def get_test_minified_transforms_torch():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ])
    return T.Compose([
        T.RandomResize([(1024, 512)], max_size=1333),
        normalize,
        T.AddGaussianNoise(0., .05)
    ])



def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided Ant path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train", root / 'ground-truth-train.json'),
        "train_eval": (root / "train", root / 'ground-truth-train.json'),
        "val": (root / "val", root / 'ground-truth-val.json'),
        "test": (root / "test", root / 'ground-truth-test.json'),
        "test_minified": (root / "test_minified", root / 'ground-truth-test_minified.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = AntDetection(img_folder, ann_file, return_masks=args.masks, image_set='val')
    return dataset
