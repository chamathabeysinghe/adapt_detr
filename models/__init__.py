# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .discriminator import build_discriminator as build_ds


def build_model(args):
    return build(args)


def build_discriminator(args):
    return build_ds(args)
