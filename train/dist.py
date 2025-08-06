#import os

#def is_master():
#    if int(os.environ["RANK"]) == 0:
#        return True
#    else:
#        return False

import os
import torch.distributed as dist

def is_distributed():
    """检查是否已初始化分布式进程组"""
    return dist.is_available() and dist.is_initialized()
def is_master():
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    else:
        # 单卡模式默认为主进程
        return True