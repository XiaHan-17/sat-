import os
import random
import datetime  # 补充datetime模块导入

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import torch.distributed as dist

from data.build_dataset import build_dataset

from model.build_model import build_maskformer, load_checkpoint, inherit_knowledge_encoder
from model.text_encoder import Text_Encoder

from train.params import parse_args
from train.logger import set_up_log
from train.loss import BinaryDiceLoss
from train.scheduler import cosine_lr
from train.trainer import Trainer
from train.dist import is_master


def set_seed(config):
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 确保可复现性（关闭benchmark，启用deterministic）
    cudnn.benchmark = False
    cudnn.deterministic = True
    # 为所有GPU设置种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # 获取配置参数
    args = parse_args()

    # 设置日志
    if is_master():
        checkpoint_dir, tb_writer, log_file = set_up_log(args)
    else:
        checkpoint_dir = None
        tb_writer = None
        log_file = None

    # 设置随机种子确保可复现性（启用种子设置）
    set_seed(args)

    # 设备和分布式环境配置
    if args.distributed:
        # 分布式模式初始化
        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # 从环境变量获取本地_rank（torchrun自动设置）
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        gpu_id = local_rank
        # 初始化分布式进程组
        torch.distributed.init_process_group(
            backend="nccl",
            init_method='env://',
            timeout=datetime.timedelta(seconds=10800)  # 延长超时时间避免训练中断
        )
    else:
        # 单卡模式配置
        if args.gpu:
            gpu_id = int(args.gpu)
        else:
            gpu_id = 0  # 默认使用第0号GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

    # 打印设备信息
    if is_master():
        print(f'** GPU NUM ** : {torch.cuda.device_count()}')
    if args.distributed:
        rank = dist.get_rank()
        if is_master():
            print(f'** WORLD SIZE ** : {torch.distributed.get_world_size()}')
        print(f"** DDP ** : Start running DDP on rank {rank}.")
    else:
        print(f"** Single GPU ** : Using GPU {gpu_id}")

    # 构建数据集和数据加载器（通过build_dataset统一处理，不再重复构建）
    trainset, trainloader, sampler = build_dataset(args)

    # 构建模型（根据分布式模式决定是否用DDP包装）
    model = build_maskformer(args, device, gpu_id)
    if args.distributed:
        # 转换为同步BN（分布式训练需要）
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpu_id],
            find_unused_parameters=True  # 允许存在未使用的参数
        )

    # 构建文本编码器
    text_encoder = Text_Encoder(
        text_encoder=args.text_encoder,
        checkpoint=args.text_encoder_checkpoint,
        partial_load=args.text_encoder_partial_load,
        biolord_checkpoint=args.biolord_checkpoint,  # 新增此行
        open_bert_layer=args.open_bert_layer,
        open_modality_embed=args.open_modality_embed,
        gpu_id=gpu_id,
        device=device
    )
    # 分布式模式下包装文本编码器
    if args.distributed:
        text_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(text_encoder)
        text_encoder = torch.nn.parallel.DistributedDataParallel(
            text_encoder,
            device_ids=[gpu_id],
            find_unused_parameters=True
        )

    # 设置损失函数
    dice_loss = BinaryDiceLoss(reduction='none')
    bce_w_logits_loss = nn.BCEWithLogitsLoss(reduction='none')  # 兼容自动混合精度

    # 设置优化器
    target_parameters = list(model.parameters()) + list(text_encoder.parameters())
    optimizer = optim.AdamW(
        target_parameters,
        lr=args.lr[0],
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    # 设置学习率调度器
    total_steps = args.step_num
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    # 处理多阶段学习率的情况
    if isinstance(total_steps, list):
        total_steps = sum(total_steps)

    # 加载检查点或预训练权重
    start_step = 1
    if args.checkpoint:
        model, optimizer, start_step = load_checkpoint(
            checkpoint=args.checkpoint,
            resume=args.resume,
            partial_load=args.partial_load,
            model=model,
            optimizer=optimizer,
            device=device,
        )
    elif args.inherit_knowledge_encoder:
        model = inherit_knowledge_encoder(
            knowledge_encoder_checkpoint=args.knowledge_encoder_checkpoint,
            model=model,
            device=device,
        )
    if is_master():
        print(f'Starting from step {start_step}')

    # 打印冻结的参数（仅主进程）
    if is_master():
        print('The following parameters in SAT are frozen:')
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(name)
        print('The following parameters in text encoder are frozen:')
        for name, param in text_encoder.named_parameters():
            if not param.requires_grad:
                print(name)

    # 初始化训练器
    trainer = Trainer(
        args=args,
        model=model,
        text_encoder=text_encoder,
        device=device,
        trainset=trainset,
        trainloader=trainloader,
        sampler=sampler,
        dice_loss=dice_loss,
        bce_w_logits_loss=bce_w_logits_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        tb_writer=tb_writer,
        checkpoint_dir=checkpoint_dir,
        log_file=log_file
    )

    # 开始训练循环
    for step in range(start_step, total_steps + 1):
        # 定期打印训练进度（仅主进程）
        if is_master() and step % 10 == 0:
            print(f'Training Step {step}')

        # 梯度累积
        for accum in range(args.accumulate_grad_interval):
            trainer.train_one_step(step)

    # 训练结束后清理分布式环境
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()