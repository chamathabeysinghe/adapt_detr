# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
import discriminator


def build_model(args):
    return build(args)


def build_discriminator(args):
    return discriminator.build_discriminator(args)
