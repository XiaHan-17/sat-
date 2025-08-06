import torch
import torch.nn as nn
import time
import os
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

from .maskformer import Maskformer

from train.dist import is_master


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def build_maskformer(args, device, gpu_id):
    model = Maskformer(args.vision_backbone, args.crop_size, args.patch_size, args.deep_supervision)
    model = model.to(device)
    
    # 仅在分布式模式下使用DDP和SyncBatchNorm
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)        
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
    
    # 打印模型参数信息（仅主进程）
    if is_master() or not args.distributed:
        print(f"** MODEL ** {get_parameter_number(model)['Total']/1e6}M parameters")
        
    return model


def load_checkpoint(
        checkpoint,
        resume,
        partial_load,
        model,
        device,
        optimizer=None,
):
    if is_master():
        print(f'** CHECKPOINT ** : Load checkpoint from {checkpoint}')

    # 安全加载 checkpoint，避免执行恶意代码
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)

    # 校验 checkpoint 结构
    if 'model_state_dict' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict' key")

    # 移除 DDP 保存的参数中的 'module.' 前缀，兼容单卡/分布式加载
    checkpoint['model_state_dict'] = {
        k[len('module.'):] if k.startswith('module.') else k: v
        for k, v in checkpoint['model_state_dict'].items()
    }

    if partial_load:
        model_dict = model.state_dict()
        # 检查参数差异
        unexpected_keys = [k for k in checkpoint['model_state_dict'] if k not in model_dict]
        missing_keys = [k for k in model_dict if k not in checkpoint['model_state_dict']]
        mismatched_shape_keys = [
            k for k, v in checkpoint['model_state_dict'].items()
            if k in model_dict and v.shape != model_dict[k].shape
        ]
        # 筛选可加载的参数
        loadable_state_dict = {
            k: v for k, v in checkpoint['model_state_dict'].items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(loadable_state_dict)
        model.load_state_dict(model_dict)

        if is_master():
            print(
                f'The following parameters are unexpected in SAT checkpoint:\n{unexpected_keys[:5]}...' if unexpected_keys else 'No unexpected parameters')
            print(
                f'The following parameters are missing in SAT checkpoint:\n{missing_keys[:5]}...' if missing_keys else 'No missing parameters')
            print(
                f'The following parameters have mismatched shapes:\n{mismatched_shape_keys[:5]}...' if mismatched_shape_keys else 'No shape mismatches')
            print(f'Successfully loaded {len(loadable_state_dict)} parameters')
    else:
        # 严格模式加载（不匹配时会报错）
        model.load_state_dict(checkpoint['model_state_dict'])
        if is_master():
            print('Loaded full model state dict in strict mode')

    # 恢复训练状态
    start_step = 1
    if resume:
        if optimizer is None:
            raise ValueError("optimizer must be provided when resume=True")
        if 'optimizer_state_dict' not in checkpoint or 'step' not in checkpoint:
            raise KeyError("Checkpoint missing 'optimizer_state_dict' or 'step' for resume")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = int(checkpoint['step']) + 1
        if is_master():
            print(f'Resumed training from step {start_step - 1}')

    return model, optimizer, start_step


def inherit_knowledge_encoder(knowledge_encoder_checkpoint,
                              model,
                              device
                              ):
    # inherit unet encoder and multiscale feature projection layer from knowledge encoder
    checkpoint = torch.load(knowledge_encoder_checkpoint, map_location=device)
        
    model_dict =  model.state_dict()
    visual_encoder_state_dict = {k.replace('atlas_tower', 'backbone'):v for k,v in checkpoint['model_state_dict'].items() if 'atlas_tower.encoder' in k}    # encoder部分
    model_dict.update(visual_encoder_state_dict)
    proj_state_dict = {k.replace('atlas_tower.', ''):v for k,v in checkpoint['model_state_dict'].items() if 'atlas_tower.projection_layer' in k}    # projection layer部分
    model_dict.update(proj_state_dict)
    model.load_state_dict(model_dict)
    
    if is_master():
        print('** CHECKPOINT ** : Inherit pretrained unet encoder from %s' % (knowledge_encoder_checkpoint))
        print('The following parameters are loaded in SAT:\n', list(visual_encoder_state_dict.keys())+list(proj_state_dict.keys()))
        
    return model