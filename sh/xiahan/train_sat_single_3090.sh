#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# export NCCL_DEBUG=INFO

python train.py \
--log_dir 'log/single_3090' \
--name 'sat_nano_3090' \
--vision_backbone 'UNET' \
--deep_supervision True \
--save_large_interval 50 \
--save_small_interval 50 \
--log_step_interval 50 \
--step_num 150 \
--warmup 50 \
--lr 5e-5 \
--accumulate_grad_interval 4 \
--datasets_jsonl '/root/autodl-tmp/SAT-DS-main/data/hugface_huangyc/task001_OAIZIB-CM/train/train_output.jsonl' \
--dataset_config '/root/autodl-tmp/SAT-DS-main/data/hugface_huangyc/task001_OAIZIB-CM/train/1.json' \
--text_encoder 'ours' \
--text_encoder_checkpoint '/root/autodl-tmp/SAT-main/sh/nano_text_encoder.pth' \
--biolord_checkpoint '/root/autodl-tmp/SAT-main/model/BioLORD-2023-C' \
--num_workers 8 \
--max_queries 16 \
--crop_size 224 224 64 \
--patch_size 32 32 32 \
--batchsize_3d 1 \
--allow_repeat True \
--gpu 0 \
--distributed False