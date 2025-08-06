#!/bin/bash
#SBATCH --job-name=sat_pro          # 任务名称，用于识别任务，此处为"sat_pro"
#SBATCH --partition=medai           # 指定提交任务的分区（队列），此处为"medai"
#SBATCH --nodes=2                   # 申请的节点数量，此处为2个节点
#SBATCH --ntasks=2                  # 总任务数，与节点数对应（每个节点1个任务）
#SBATCH --gpus-per-task=8           # 每个任务分配的GPU数量，此处每个任务8张GPU
#SBATCH --cpus-per-task=16          # 每个任务分配的CPU核心数，此处16核
#SBATCH --mem-per-cpu=256G          # 每个CPU核心分配的内存，此处256GB
#SBATCH --chdir=logs                # 任务运行的工作目录，此处为"logs"文件夹
#SBATCH --output=logs/%x-%j.out     # 标准输出日志路径，%x为任务名，%j为任务ID
#SBATCH --error=logs/%x-%j.error    # 错误输出日志路径，同上
###SBATCH --exclude=xxx              # 注释掉的参数，用于排除特定节点（如需启用可删除###）

export NCCL_DEBUG=INFO              # 启用NCCL（NVIDIA集体通信库）的调试日志，输出INFO级别信息
export NCCL_IBEXT_DISABLE=1         # 禁用IB（InfiniBand）扩展，避免通信冲突
export NCCL_IB_DISABLE=1            # 禁用IB通信，强制使用TCP/IP
export NCCL_SOCKET_IFNAME=eth0      # 指定用于NCCL通信的网络接口为eth0
echo NODELIST=${SLURM_NODELIST}     # 打印任务分配的节点列表
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  # 获取主节点地址（第一个节点）
export MASTER_ADDR=$master_addr     # 设置主节点地址为环境变量，供分布式训练使用
MASTER_PORT=$((RANDOM % 101 + 20000))  # 随机生成一个20000-20100之间的端口作为主节点通信端口
echo "MASTER_ADDR="$MASTER_ADDR     # 打印主节点地址

srun torchrun \
--nnodes 2 \                        # 分布式训练的节点总数，与Slurm的--nodes一致
--nproc_per_node 8 \                # 每个节点上启动的进程数（即使用的GPU数），与--gpus-per-task一致
--rdzv_id 100 \                     # 分布式训练的唯一标识ID，用于节点发现
--rdzv_backend c10d \               # 分布式后端，使用PyTorch的c10d
--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py \    # 主节点的地址和端口，用于进程通信
--log_dir 'log' \                   # 日志和 checkpoint 保存的根目录
--name 'sat_pro' \                  # 实验名称，用于区分不同训练任务
--vision_backbone 'UNET-L' \        # 视觉骨干网络类型，此处为大型UNET（UNET-L）
--deep_supervision True \           # 是否启用深度监督（使用中间层输出辅助训练）
--save_large_interval 10000 \       # 保存大型checkpoint的间隔（每10000步）
--save_small_interval 1000 \        # 保存小型checkpoint的间隔（每1000步，用于中断恢复）
--log_step_interval 1000 \          # 日志输出间隔（每1000步打印一次训练信息）
--step_num 200000 \                 # 总训练步数，共200000步
--warmup 20000 \                    # 学习率热身步数，前20000步从0线性增长到目标学习率
--lr 1e-4 \                         # 目标学习率，为1e-4
--accumulate_grad_interval 1 \      # 梯度累积间隔，每1步更新一次参数（不累积）
--datasets_jsonl 'trainset.jsonl' \ # 训练数据集的JSONL文件路径，包含样本信息
--dataset_config 'data/dataset_config/72.json' \  # 数据集配置文件路径，定义数据增强等参数
--text_encoder 'ours' \             # 文本编码器类型，使用自定义的编码器
--text_encoder_checkpoint 'checkpoint/text_encoder_checkpoint.pth' \  # 文本编码器的预训练权重路径
--text_encoder_partial_load False \ # 是否部分加载文本编码器权重（False表示全量加载）
--open_bert_layer 12 \              # 开放BERT的前12层参数用于训练（即解冻这些层）
--open_modality_embed True \        # 是否开放模态嵌入层参数用于训练
--num_workers 8 \                   # 数据加载的进程数，每个进程处理8个worker
--max_queries 32 \                  # 每个样本的最大查询数（文本提示数量）
--crop_size 288 288 64 \            # 输入图像的裁剪尺寸（H, W, D）为288x288x64
--patch_size 32 32 32 \             # 图像分块的尺寸（H, W, D）为32x32x32
--batchsize_3d 1 \                  # 3D图像的批次大小，每个GPU每次处理1个样本
--allow_repeat True \               # 是否允许重复采样标签较多的样本（加速收敛）