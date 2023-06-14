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

from core.model.model import GAA
from core.loss.combine_loss import CombineLoss
from interface.argparse import get_args
from utils.data_loader import ImageDataset
from utils.engine import train_one_epoch, eval_one_epoch


def main():

    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            f'[Info] You have chosen specific seed : {args.seed}'
            )
    
    if args.gpu is not None:
        warnings.warn(
            f'[Info] You have chosen specific GPU. This will be completely disable data parallelism.'
        )

    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:

        args.world_size = ngpus_per_node * args.world_size

        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args)
        )
    else:

        main_worker(
            args.gpu,
            ngpus_per_node,
            args
        )

def main_worker(
        gpu: object,
        ngpus_per_node: int,
        args: object
):
    
    writer = SummaryWriter(args.save_writer)

    args.gpu = gpu

    if args.distributed:

        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu

        print(
            f'\n[GPU INFO] dist_backend : {args.dist_backend}'
            )
        print(
            f'[GPU INFO] world_size : {args.world_size}'
            )
        print(
            f'[GPU INFO] rank : {args.rank}'
        )

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

    # get training dataset 
    train_dataset=ImageDataset(
        data_path=args.train_data_path,
        scaling=True,
        class_converter=False,
        label_converter=True
    )

    train_sampler = DistributedSampler(train_dataset)

    # load training dataset into device(s)
    train_data = DataLoader(
        train_dataset, 
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=None,
        shuffle=False,
        num_workers=args.num_workers
    )

    # get validation dataset
    validation_dataset=ImageDataset(
        data_path=args.val_data_path,
        scaling=True,
        class_converter=False,
        label_converter=True
    )

    # load validation dataset into devices(s)
    validation_sampler = DistributedSampler(validation_dataset)

    validation_data = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        batch_size=args.batch_size,
        collate_fn=None,
        shuffle=False,
        num_workers=args.num_workers
    )

    # load model(s) into device(s)
    machine = GAA(
        image_size=512, patch_size=8, 
        num_classes=8, in_channels=1
    )

    if not torch.cuda.is_available():
        print(
            f'[INFO] Using CPU'
            )
    
    elif args.distributed:
        if args.gpu is not None:

            torch.cuda.set_device(args.gpu)
            machine.cuda(args.gpu)

            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node -1) / ngpus_per_node)

            machine = DDP(
                machine,
                device_ids=[args.gpu],
                find_unused_parameters=True
            )

            if args.gpu == 0:
                print(
                    f'\n[INFO] Using DDP with sepcific GPUs '
                )
        
        else:

            machine.cuda()

            machine = DDP(machine)
            print(
                f'\n[INFO] Using DDP with single GPUs'
            )

    else:
        machine = torch.nn.DataParallel(machine).cuda() 

    # set optimizer, scheduler and criterion
    optimizer = Adam(machine.parameters(),
                              lr=args.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = CombineLoss()

    for epoch in range(1, args.epochs):

        if args.gpu is not None:
            train_sampler.set_epoch(epoch)
            validation_sampler.set_epoch(epoch)
        
        train_one_epoch(
            machine=machine, train_loader=train_data, 
            optimizer=optimizer, criterion=criterion, 
            args=args, epoch=epoch, writer=writer
        )
        eval_one_epoch(
            machine=machine, val_loader=validation_data, 
            scheduler=scheduler, criterion=criterion, 
            args=args, epoch=epoch, writer=writer
        )


if __name__ == '__main__':

    main()
    