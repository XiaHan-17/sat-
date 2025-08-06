#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py \
--log_dir 'log/finetune_single_3090' \
--name 'sat_nano_finetune' \
--vision_backbone 'UNET' \
--deep_supervision True \
--save_large_interval 4050 \
--save_small_interval 2025 \
--log_step_interval 810 \
--step_num 4050 \
--warmup 50 \
--lr 1e-5 \
--accumulate_grad_interval 4 \
--datasets_jsonl '/root/autodl-tmp/SAT-DS-main/data/hugface_huangyc/task001_OAIZIB-CM/train/train_output.jsonl' \
--dataset_config '/root/autodl-tmp/SAT-DS-main/data/hugface_huangyc/task001_OAIZIB-CM/train/1.json' \
--text_encoder 'ours' \
--text_encoder_checkpoint '/root/autodl-tmp/SAT-main/demo/inference_demo/nano_text_encoder.pth' \
--text_encoder_partial_load False \
--biolord_checkpoint '/root/autodl-tmp/SAT-main/model/BioLORD-2023-C' \
--checkpoint '/root/autodl-tmp/SAT-main/demo/inference_demo/nano.pth' \
--resume False \
--partial_load False \
--open_bert_layer 6 \
--open_modality_embed True \
--num_workers 8 \
--max_queries 16 \
--crop_size 224 224 64 \
--patch_size 32 32 32 \
--batchsize_3d 1 \
--allow_repeat True \
--gpu 0 \
--distributed False