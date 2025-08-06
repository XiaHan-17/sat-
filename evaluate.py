import os
import time
import pickle
import shutil

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import gaussian_filter
import torch.distributed as dist
import csv

from evaluate.metric import calculate_metric_percase
from train.dist import is_master, is_distributed
from data.evaluate_dataset import Evaluate_Dataset, Evaluate_Dataset_OnlineCrop, collate_fn
from model.build_model import build_maskformer, load_checkpoint
from model.text_encoder import Text_Encoder
from evaluate.params import parse_args  # 你的参数解析脚本


def compute_gaussian(tile_size, sigma_scale: float = 1. / 8, value_scaling_factor: float = 10, dtype=np.float16):
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # 避免0值产生NaN
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def evaluate(model,
             text_encoder,
             device,
             testset,
             testloader,
             dice_score,
             nsd_score,
             csv_path,
             resume,
             save_interval,
             visualization):
    # 可视化保存路径
    if visualization:
        nib_dir = csv_path.replace('.csv', '')

    # 主进程初始化，复制jsonl文件并准备结果收集结构
    if is_master():
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        shutil.copy(testset.jsonl_file, csv_path.replace('.csv', '.jsonl'))

    results_of_samples = []  # 存储所有样本的评估结果

    # 恢复中断的评估（仅主进程）
    if resume and is_master():
        root_dir = os.path.dirname(csv_path)
        prefix = os.path.basename(csv_path).replace('.csv', '_tmp_rank')
        pkl_to_del = []
        for f in os.listdir(root_dir):
            if prefix in f:
                pkl_path = os.path.join(root_dir, f)
                with open(pkl_path, 'rb') as ftmp:
                    results_of_samples.extend(pickle.load(ftmp))
                pkl_to_del.append(pkl_path)
        for pkl_path in pkl_to_del:
            os.remove(pkl_path)

        # 合并成一个统一的临时文件，方便后续读取
        merge_pkl = csv_path.replace('.csv', '_tmp_rank0.pkl')
        with open(merge_pkl, 'wb') as fmerge:
            pickle.dump(results_of_samples, fmerge)
        print(f'恢复已加载 {len(results_of_samples)} 个样本结果，合并到 {merge_pkl}')

    model.eval()
    text_encoder.eval()

    gaussian = torch.tensor(compute_gaussian((288, 288, 96))).to(device)  # 高斯核，用于平滑叠加

    if is_master():
        testloader = tqdm(testloader, disable=False)
    else:
        testloader = tqdm(testloader, disable=True)

    end_time = time.time()

    for sample in testloader:
        dataset_name = sample['dataset_name']
        sample_id = sample['sample_id']
        batched_patches = sample['batched_patches']
        batched_y1y2_x1x2_z1z2 = sample['batched_y1y2_x1x2_z1z2']
        split_labels = sample['split_labels']
        split_n1n2 = sample['split_n1n2']
        gt_segmentation = sample['gt_segmentation'].numpy()
        labels = sample['labels']
        modality = sample['modality']
        image_path = sample['image_path']

        n, h, w, d = gt_segmentation.shape
        prediction = torch.zeros((n, h, w, d))
        accumulation = torch.zeros((n, h, w, d))

        with autocast():
            queries_ls = []
            for labels_ls, n1n2 in zip(split_labels, split_n1n2):
                queries_ls.append(text_encoder(labels_ls, modality))

            for patches, y1y2_x1x2_z1z2_ls in zip(batched_patches, batched_y1y2_x1x2_z1z2):
                patches = patches.to(device=device)
                prediction_patch = model(queries=queries_ls, image_input=patches)
                prediction_patch = torch.sigmoid(prediction_patch).detach()

                for b in range(len(y1y2_x1x2_z1z2_ls)):
                    y1, y2, x1, x2, z1, z2 = y1y2_x1x2_z1z2_ls[b]
                    tmp = prediction_patch[b, :, :y2 - y1, :x2 - x1, :z2 - z1] * gaussian[:y2 - y1, :x2 - x1, :z2 - z1]
                    prediction[:, y1:y2, x1:x2, z1:z2] += tmp.cpu()
                    accumulation[:, y1:y2, x1:x2, z1:z2] += gaussian[:y2 - y1, :x2 - x1, :z2 - z1].cpu()

        prediction = prediction / accumulation
        prediction = torch.where(prediction > 0.5, 1.0, 0.0).numpy()

        scores = []
        for j in range(len(labels)):
            scores.append(calculate_metric_percase(prediction[j], gt_segmentation[j], dice_score, nsd_score))

        # 可视化保存
        if visualization:
            Path(f'{nib_dir}/{dataset_name}').mkdir(exist_ok=True, parents=True)
            results = np.zeros((h, w, d))
            for j, label in enumerate(labels):
                results += prediction[j] * (j + 1)
                Path(f'{nib_dir}/{dataset_name}/seg_{sample_id}').mkdir(exist_ok=True, parents=True)
                nib.save(nib.nifti2.Nifti1Image(prediction[j], np.eye(4)),
                         f'{nib_dir}/{dataset_name}/seg_{sample_id}/{label}.nii.gz')
            nib.save(nib.nifti2.Nifti1Image(results, np.eye(4)), f'{nib_dir}/{dataset_name}/seg_{sample_id}.nii.gz')

            image = testset.load_image(image_path)
            image = np.squeeze(image)
            nib.save(nib.nifti2.Nifti1Image(image, np.eye(4)), f'{nib_dir}/{dataset_name}/img_{sample_id}.nii.gz')

            gt = np.zeros((h, w, d))
            for j, label in enumerate(labels):
                gt += gt_segmentation[j] * (j + 1)
                Path(f'{nib_dir}/{dataset_name}/gt_{sample_id}').mkdir(exist_ok=True, parents=True)
                nib.save(nib.nifti2.Nifti1Image(gt_segmentation[j], np.eye(4)),
                         f'{nib_dir}/{dataset_name}/gt_{sample_id}/{label}.nii.gz')
            nib.save(nib.nifti2.Nifti1Image(gt, np.eye(4)), f'{nib_dir}/{dataset_name}/gt_{sample_id}.nii.gz')

        results_of_samples.append([dataset_name, modality, sample_id, scores, labels])

        # 定期保存临时结果
        if len(results_of_samples) % save_interval == 0:
            current_rank = dist.get_rank() if is_distributed() else 0
            tmp_path = csv_path.replace('.csv', f'_tmp_rank{current_rank}.pkl')
            with open(tmp_path, 'wb') as ftmp:
                pickle.dump(results_of_samples, ftmp)

    # 评估循环结束，主进程合并所有临时结果，写出csv和txt
    if is_master():
        root_dir = os.path.dirname(csv_path)
        prefix = os.path.basename(csv_path).replace('.csv', '_tmp_rank')
        all_results = []

        # 合并所有临时pkl文件
        for f in os.listdir(root_dir):
            if prefix in f and f.endswith('.pkl'):
                tmp_path = os.path.join(root_dir, f)
                with open(tmp_path, 'rb') as ftmp:
                    all_results.extend(pickle.load(ftmp))
                os.remove(tmp_path)
                print(f"合并并删除临时文件：{f}")

        # 写入CSV文件
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Dataset', 'Modality', 'SampleID', 'Label', 'Dice', 'NSD'])
            for sample in all_results:
                dataset_name, modality, sample_id, scores, labels = sample
                for label, score in zip(labels, scores):
                    dice_val = score.get('dice', '') if score else ''
                    nsd_val = score.get('nsd', '') if score else ''
                    writer.writerow([dataset_name, modality, sample_id, label, dice_val, nsd_val])
        print(f"最终结果写入CSV文件: {csv_path}")

        # 写入TXT文件
        txt_path = csv_path.replace('.csv', '.txt')
        with open(txt_path, 'w') as ftxt:
            for sample in all_results:
                dataset_name, modality, sample_id, scores, labels = sample
                ftxt.write(f"Sample {dataset_name} {sample_id} ({modality}):\n")
                for label, score in zip(labels, scores):
                    dice_val = score.get('dice', '') if score else ''
                    nsd_val = score.get('nsd', '') if score else ''
                    ftxt.write(f"  {label}: Dice={dice_val}, NSD={nsd_val}\n")
                ftxt.write('\n')
        print(f"最终结果写入TXT文件: {txt_path}")


def main(args):
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if args.distributed:
        dist.init_process_group(backend='nccl', init_method='env://', timeout=torch.timedelta(seconds=10800))

    evaluated_samples = set()
    if args.resume:
        pass  # 可加入你自己的恢复逻辑

    if args.online_crop:
        testset = Evaluate_Dataset_OnlineCrop(args.datasets_jsonl, args.max_queries, args.batchsize_3d, args.crop_size,
                                              evaluated_samples)
    else:
        testset = Evaluate_Dataset(args.datasets_jsonl, args.max_queries, args.batchsize_3d, args.crop_size,
                                   evaluated_samples)

    sampler = torch.utils.data.distributed.DistributedSampler(testset) if args.distributed else None
    testloader = torch.utils.data.DataLoader(testset, sampler=sampler, batch_size=1, pin_memory=args.pin_memory,
                                             num_workers=args.num_workers, collate_fn=collate_fn, shuffle=False)
    if sampler:
        sampler.set_epoch(0)

    model = build_maskformer(args, device, local_rank)

    text_encoder = Text_Encoder(
        text_encoder=args.text_encoder,
        checkpoint=args.text_encoder_checkpoint,
        partial_load=args.partial_load,
        open_bert_layer=12,
        open_modality_embed=False,
        gpu_id=local_rank,
        device=device,
        biolord_checkpoint=args.biolord_checkpoint
    )

    model, _, _ = load_checkpoint(
        checkpoint=args.checkpoint,
        resume=False,
        partial_load=args.partial_load,
        model=model,
        device=device
    )

    evaluate(model=model,
             text_encoder=text_encoder,
             device=device,
             testset=testset,
             testloader=testloader,
             dice_score=args.dice,
             nsd_score=args.nsd,
             csv_path=f'{args.rcd_dir}/{args.rcd_file}.csv',
             resume=args.resume,
             save_interval=args.save_interval,
             visualization=args.visualization)


if __name__ == '__main__':
    args = parse_args()
    main(args)

