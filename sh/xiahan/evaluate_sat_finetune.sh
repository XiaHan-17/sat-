#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


python evaluate.py \
--rcd_dir '/root/autodl-tmp/SAT-main/log/eval_results' \
--rcd_file 'sat_pro_eval_3090' \
--resume False \
--visualization False \
--deep_supervision False \
--datasets_jsonl '/root/autodl-tmp/SAT-DS-main/data/hugface_huangyc/task001_OAIZIB-CM/train/test_output.jsonl' \
--crop_size 224 224 64 \
--online_crop True \
--vision_backbone 'UNET' \
--checkpoint '/root/autodl-tmp/SAT-main/log/finetune_single_3090/sat_nano_finetune/checkpoint/latest_step.pth' \
--partial_load True \
--text_encoder 'ours' \
--text_encoder_checkpoint '/root/autodl-tmp/SAT-main/log/finetune_single_3090/sat_nano_finetune/checkpoint/text_encoder_latest_step.pth' \
--biolord_checkpoint '/root/autodl-tmp/SAT-main/model/BioLORD-2023-C' \
--batchsize_3d 1 \
--max_queries 256 \
--pin_memory False \
--num_workers 8 \
--dice True \
--nsd True \
--gpu 0 \
--save_interval 10 \
--distributed False