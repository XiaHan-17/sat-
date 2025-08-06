import os
import time

import torch
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
from einops import rearrange, repeat, reduce
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
import shutil
import pickle
from scipy.ndimage import gaussian_filter
import torch.distributed as dist

from evaluate.metric import calculate_metric_percase
from evaluate.merge_after_evaluate import merge
from train.dist import is_master, is_distributed  # 新增is_distributed导入


def compute_gaussian(tile_size, sigma_scale: float = 1. / 8, value_scaling_factor: float = 10, dtype=np.float16):
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # 确保没有零值，避免NaN
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

    # 主进程处理结果汇总
    if is_master():
        shutil.copy(testset.jsonl_file, csv_path.replace('.csv', '.jsonl'))

        # 数据集-标签-指标映射
        datasets_labels_metrics = {}
        # 样本-标签-指标映射
        samples_labels_metrics = {}
        # 数据集-标签集合映射
        datasets_labels_sets = {}

    # 累积每个样本的结果
    results_of_samples = []  # 元素格式: [dataset_name, modality, sample_id, scores_of_labels(dict), label_names]

    # 恢复中断的评估（仅主进程）
    if resume and is_master():
        root_dir = os.path.dirname(csv_path)
        prefix = os.path.basename(csv_path).replace('.csv', '_tmp_rank')
        pkl_to_del = []
        for f in os.listdir(root_dir):
            if prefix in f:
                pkl_path = f'{root_dir}/{f}'
                with open(pkl_path, 'rb') as f:
                    results_of_samples += pickle.load(f)
                print(f'从 {pkl_path} 加载结果')
                pkl_to_del.append(pkl_path)

        # 去重并合并到临时文件
        for pkl_path in pkl_to_del:
            os.remove(pkl_path)
            print(f'删除 {pkl_path}')
        merge_pkl = csv_path.replace('.csv', f'_tmp_rank0.pkl')
        with open(merge_pkl, 'wb') as f:
            pickle.dump(results_of_samples, f)
        print(f'已加载 {len(results_of_samples)} 个样本结果，合并到 {merge_pkl}')

    model.eval()
    text_encoder.eval()

    with torch.no_grad():
        data_time = 0
        pred_time = 0
        metric_time = 0

        avg_patch_batch_num = 0
        avg_query_batch_num = 0

        # 仅主进程显示进度条
        if is_master():
            testloader = tqdm(testloader, disable=False)
        else:
            testloader = tqdm(testloader, disable=True)

        # 用于累积预测的高斯核
        gaussian = torch.tensor(compute_gaussian((288, 288, 96))).to(device)  # hwd

        end_time = time.time()
        for sample in testloader:  # 评估时每个batch是一个体素
            # 数据加载
            dataset_name = sample['dataset_name']
            sample_id = sample['sample_id']
            batched_patches = sample['batched_patches']
            batched_y1y2_x1x2_z1z2 = sample['batched_y1y2_x1x2_z1z2']
            split_labels = sample['split_labels']
            split_n1n2 = sample['split_n1n2']
            gt_segmentation = sample['gt_segmentation'].numpy()  # n h w d
            labels = sample['labels']
            modality = sample['modality']
            image_path = sample['image_path']

            n, h, w, d = gt_segmentation.shape
            prediction = torch.zeros((n, h, w, d))
            accumulation = torch.zeros((n, h, w, d))

            data_time += (time.time() - end_time)
            end_time = time.time()

            avg_patch_batch_num += len(batched_patches)
            avg_query_batch_num += len(split_labels)

            with autocast():
                # 处理文本查询
                queries_ls = []
                for labels_ls, n1n2 in zip(split_labels, split_n1n2):
                    queries_ls.append(text_encoder(labels_ls, modality))

                # 处理图像补丁
                for patches, y1y2_x1x2_z1z2_ls in zip(batched_patches, batched_y1y2_x1x2_z1z2):  # [b, c, h, w, d]
                    patches = patches.to(device=device)
                    prediction_patch = model(queries=queries_ls, image_input=patches)
                    prediction_patch = torch.sigmoid(prediction_patch)  # bnhwd
                    prediction_patch = prediction_patch.detach()

                    # 填充预测结果
                    for b in range(len(y1y2_x1x2_z1z2_ls)):
                        y1, y2, x1, x2, z1, z2 = y1y2_x1x2_z1z2_ls[b]

                        # 高斯加权累积
                        tmp = prediction_patch[b, :, :y2 - y1, :x2 - x1, :z2 - z1] * gaussian[:y2 - y1, :x2 - x1,
                                                                                     :z2 - z1]  # GPU上计算
                        prediction[:, y1:y2, x1:x2, z1:z2] += tmp.cpu()
                        accumulation[:, y1:y2, x1:x2, z1:z2] += gaussian[:y2 - y1, :x2 - x1, :z2 - z1].cpu()

            pred_time += (time.time() - end_time)
            end_time = time.time()

            # 计算平均预测
            prediction = prediction / accumulation
            prediction = torch.where(prediction > 0.5, 1.0, 0.0)
            prediction = prediction.numpy()

            # 计算指标
            scores = []
            for j in range(len(labels)):
                scores.append(calculate_metric_percase(prediction[j, :, :, :], gt_segmentation[j, :, :, :], dice_score,
                                                       nsd_score))  # 每个标签的指标字典

            # 可视化保存
            if visualization:
                Path(f'{nib_dir}/{dataset_name}').mkdir(exist_ok=True, parents=True)
                # 合并所有标签的预测结果
                results = np.zeros((h, w, d))  # hwd
                for j, label in enumerate(labels):
                    results += prediction[j, :, :, :] * (j + 1)  # 避免背景重叠
                    Path(f'{nib_dir}/{dataset_name}/seg_{sample_id}').mkdir(exist_ok=True, parents=True)
                    # 保存单个标签的预测
                    segobj = nib.nifti2.Nifti1Image(prediction[j, :, :, :], np.eye(4))
                    nib.save(segobj, f'{nib_dir}/{dataset_name}/seg_{sample_id}/{label}.nii.gz')
                # 保存合并的预测结果
                segobj = nib.nifti2.Nifti1Image(results, np.eye(4))
                nib.save(segobj, f'{nib_dir}/{dataset_name}/seg_{sample_id}.nii.gz')

                # 保存原图
                image = testset.load_image(image_path)
                image = np.squeeze(image)
                imgobj = nib.nifti2.Nifti1Image(image, np.eye(4))
                nib.save(imgobj, f'{nib_dir}/{dataset_name}/img_{sample_id}.nii.gz')

                # 处理并保存GT
                gt = np.zeros((h, w, d))  # hwd
                for j, label in enumerate(labels):
                    gt += gt_segmentation[j, :, :, :] * (j + 1)
                    Path(f'{nib_dir}/{dataset_name}/gt_{sample_id}').mkdir(exist_ok=True, parents=True)
                    # 保存单个标签的GT
                    segobj = nib.nifti2.Nifti1Image(gt_segmentation[j, :, :, :], np.eye(4))
                    nib.save(segobj, f'{nib_dir}/{dataset_name}/gt_{sample_id}/{label}.nii.gz')
                # 保存合并的GT
                gtobj = nib.nifti2.Nifti1Image(gt, np.eye(4))
                nib.save(gtobj, f'{nib_dir}/{dataset_name}/gt_{sample_id}.nii.gz')

            metric_time += (time.time() - end_time)
            end_time = time.time()

            # 累积结果
            results_of_samples.append([dataset_name, modality, sample_id, scores, labels])

            # 定期保存临时结果（兼容单卡/分布式）
            if len(results_of_samples) % save_interval == 0:
                # 获取当前进程rank（单卡模式默认为0）
                current_rank = dist.get_rank() if is_distributed() else 0
                with open(csv_path.replace('.csv', f'_tmp_rank{current_rank}.pkl'), 'wb') as f:
                    pickle.dump(results_of_samples, f)

        # 注释掉分布式聚集代码（如需启用需添加检查）
        """
        if is_distributed():
            gather_results = [None for i in range(dist.get_world_size())]
            dist.gather_object(
                results_of_samples, 
                gather_results if dist.get_rank() == 0 else None,
                dst = 0
            )
            if int(dist.get_rank()) == 0:
                results_of_samples = [tmp for ls in results_of_samples for tmp in ls]
        """