# -- coding : utf-8 -*-

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
Last modified : 2023.03.09
Code version : 1.0
"""

import os
import warnings
import random
import numpy as np
from collections import OrderedDict
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from core.model.model import GAA
from core.loss.combine_loss import CombineLoss
from interface.argparse import get_args
from utils.data_loader import ImageDataset
from utils.engine import inference

COLOR_MAP = [
[255,0,0],
[0,255,0],
[0,0,255],
[255,255,0],
[255,0,255],
[0,255,255],
[0,0,0],
[255,255,255]
]

if __name__ == '__main__':

    args = get_args()
    args.device = 'cpu'

    test_dataset=ImageDataset(
        data_path=args.val_data_path,
        scaling=True,
        class_converter=False,
        label_converter=True
    )

    test_data = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=None,
        shuffle=False,
        num_workers=args.num_workers
    )

    machine = GAA(
        image_size=512, patch_size=8, 
        num_classes=8, in_channels=1
    )

    weights = torch.load('./logs/400_v4.pt', map_location=args.device)

    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k[7:]
        new_state_dict[name] = v

    machine.load_state_dict(new_state_dict)

    criterion = CombineLoss()
    
    inference(
        machine=machine, val_loader=test_data, 
        criterion=criterion, args=args
    )
    