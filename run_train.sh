#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun train.py --multiprocessing_distributed --rank 0 --batch_size 4 --epochs 40000 --save_model_path './logs/' --world_size 1 --train_data_path '../13_DataBuilder/my_data_v_3_1.hdf5' --val_data_path '../13_DataBuilder/my_data_v_3_1_test.hdf5' --n_classes 8 --learning_rate 0.00001
