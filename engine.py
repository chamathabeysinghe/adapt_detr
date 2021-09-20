# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    discriminator_model: torch.nn.Module, discriminator_criterion: torch.nn.Module,
                    data_loader: Iterable, data_loader_val_iter,
                    optimizer: torch.optim.Optimizer, discriminator_optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, gan_loss_coef: float, batch_size: int, max_norm: float = 0):
    model.train()
    discriminator_model.train()
    criterion.train()
    discriminator_criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        samples_val, targets_val = next(data_loader_val_iter)  # TODO Create new data generator
        samples_val = samples_val.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        N = batch_size
        true_labels = torch.ones(N).to(device)
        fake_labels = torch.zeros(N).to(device)


        # Train discriminator
        discriminator_optimizer.zero_grad()
        outputs, _, source_features = model(samples)
        source_features = source_features[-1].tensors
        _, _, target_features = model(samples_val, feature_only=True)
        target_features = target_features[-1].tensors
        discriminator_output_source = discriminator_model(source_features.detach()).view(-1)
        discriminator_loss_1 = discriminator_criterion(discriminator_output_source, true_labels)
        discriminator_loss_1.backward()

        discriminator_output_target = discriminator_model(target_features.detach()).view(-1)
        discriminator_loss_2 = discriminator_criterion(discriminator_output_target, fake_labels)
        discriminator_loss_2.backward()
        discriminator_optimizer.step()

        # Train Generator + Transformer
        discriminator_output_target_new = discriminator_model(target_features).view(-1)
        generator_loss = discriminator_criterion(discriminator_output_target_new, true_labels)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        total_loss = losses + gan_loss_coef * generator_loss

        # for logging purpose
        gan_loss_dict = {'loss_generator': generator_loss,
                         'loss_discriminator': discriminator_loss_1+discriminator_loss_2}
        gan_loss_dict_reduced = utils.reduce_dict(gan_loss_dict)
        gan_loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in gan_loss_dict_reduced.items()}
        gan_loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in gan_loss_dict_reduced.items() if k in weight_dict}


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        total_loss_value = loss_value + gan_loss_dict_reduced_scaled['loss_generator'].item()

        if not math.isfinite(total_loss_value):
            print("Loss is {}, stopping training".format(total_loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        total_loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled,
                             **gan_loss_dict_reduced_scaled, **gan_loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(total_loss=total_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, discriminator_model, discriminator_criterion, postprocessors, data_loader, base_ds, device, output_dir, batch_size):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        N = batch_size
        true_labels = torch.ones(N).to(device)
        fake_labels = torch.zeros(N).to(device)

        outputs, _, target_features = model(samples)
        target_features = target_features[-1].tensors
        discriminator_output_target_new = discriminator_model(target_features).view(-1)
        generator_loss = discriminator_criterion(discriminator_output_target_new, true_labels)
        discriminator_loss = discriminator_criterion(discriminator_output_target_new, fake_labels)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # for logging purpose
        gan_loss_dict = {'loss_generator': generator_loss, 'loss_discriminator': discriminator_loss}
        gan_loss_dict_reduced = utils.reduce_dict(gan_loss_dict)
        gan_loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in gan_loss_dict_reduced.items()}
        gan_loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in gan_loss_dict_reduced.items() if k in weight_dict}

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        losses = sum(loss_dict_reduced_scaled.values())
        total_loss_value = losses + gan_loss_dict_reduced_scaled['loss_generator']

        metric_logger.update(loss=losses,
                             total_loss=total_loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled,
                             **gan_loss_dict_reduced_scaled,
                             **gan_loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
