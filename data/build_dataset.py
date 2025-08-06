import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .train_dataset import Med_SAM_Dataset
from .collect_fn import collect_fn


def build_dataset(args):
    # 构建数据集（通用，单卡/分布式共享）
    dataset = Med_SAM_Dataset(
        jsonl_file=args.datasets_jsonl,
        crop_size=args.crop_size,  # h w d
        max_queries=args.max_queries,
        dataset_config=args.dataset_config,
        allow_repeat=args.allow_repeat
    )
    
    # 根据分布式模式选择采样器
    if args.distributed:
        # 分布式模式：使用DistributedSampler
        sampler = DistributedSampler(dataset)
    else:
        # 单卡模式：不使用分布式采样器（设为None，后续DataLoader会自动使用默认采样器）
        sampler = None
    
    # 构建数据加载器（根据是否分布式动态配置）
    if args.num_workers is not None:
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batchsize_3d,
            collate_fn=collect_fn,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            shuffle=(not args.distributed)  # 单卡模式启用shuffle，分布式模式由sampler控制shuffle
        )
    else:
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batchsize_3d,
            collate_fn=collect_fn,
            pin_memory=args.pin_memory,
            shuffle=(not args.distributed)  # 单卡模式启用shuffle
        )
    
    return dataset, dataloader, sampler