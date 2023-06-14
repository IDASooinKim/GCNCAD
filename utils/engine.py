# -- coding : utf-8 -*-

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
Last modified : 2023.03.09
Code version : 1.0
"""

import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
from pkbar import Kbar
from PIL import Image

# torch modules for ignition
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import node_classification_acc, node_classification_IoU
# user define modules for ignition
from interface.message import *


global COLOR_MAP
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


def train_one_epoch(
        machine:nn.Module, train_loader:DataLoader, 
        optimizer:torch.optim, criterion:torch.nn, 
        args:object, epoch:int, writer:SummaryWriter
    ):

    bar = Kbar(target=len(train_loader))
    total_loss = float(0)
    machine.train()
    
    if args.gpu == 0:
        print_current_epoch(epoch)
    
    for batch_iter, data in enumerate(train_loader):
        
        features, classes, labels = data

        features = features.type(torch.float32).to("cuda")
        classes = classes.type(torch.float32).to("cuda")
        labels = labels.type(torch.long).to("cuda")

        optimizer.zero_grad()
    
        classes_pred, labels_pred = machine(features)
        
        loss = criterion(
            classes_pred, classes, 
            labels_pred, labels
            )

        loss.backward(retain_graph=True)
        optimizer.step()

        if args.gpu == 0:
            print_progress(
                bar=bar, batch_iter=batch_iter,
                loss=loss.item()
            )
           
        total_loss += loss.item()

    if args.gpu == 0:
        writer.add_scalar("LOSS/Train", total_loss/len(train_loader), epoch)
        bar.add(1)


def eval_one_epoch(
        machine:nn.Module, val_loader:DataLoader, 
        scheduler:torch.nn, criterion:torch.nn, 
        args:object, epoch:int, writer:SummaryWriter
    ):

    total_val_loss = 0

    machine.eval()
    
    for data in val_loader:
        
        features, classes, labels = data

        features = features.type(torch.float32).to("cuda")
        classes = classes.type(torch.float32).to("cuda")
        labels = labels.type(torch.long).to("cuda")

        classes_pred, labels_pred = machine(features)

        loss = criterion(
            classes_pred, classes, 
            labels_pred, labels
            )

        total_val_loss += loss.item()

    scheduler.step()

    if args.gpu == 0:
        print_val(
            total_val_loss,
            len(val_loader)
        )
        #if img_idx == 0:
        pred_img, org_labels = vec2img(
            pred_imgs=labels_pred,
            gt_imgs=labels,
            input_imgs=features)
        re_img = torch.hstack((pred_img, org_labels))

        writer.add_image("Predictions",re_img, dataformats='HWC')
        #writer.add_image("Ground-truth",org_labels, dataformats='HWC')

        writer.add_scalar("LOSS/Val", total_val_loss/len(val_loader), epoch)

    if args.gpu==0:
        if epoch % 20 == 0:
            # torch.save(labels_pred.detach().cpu().numpy(), args.save_model_path + f'{epoch}_sam_v3.npy')
            torch.save(machine.state_dict(), args.save_model_path + f'{epoch}_v4.pt')


def inference(
        machine:nn.Module, val_loader:DataLoader, 
        criterion:torch.nn, args:object
    ):

    total_val_loss = 0

    machine.eval()
    
    for name, data in tqdm(enumerate(val_loader)):
        
        features, classes, labels = data

        features = features.type(torch.float32).to(args.device)
        classes = classes.type(torch.float32).to(args.device)
        labels = labels.type(torch.long).to(args.device)

        classes_pred, labels_pred = machine(features)

        pred_img, org_labels = vec2img(
            pred_imgs=labels_pred,
            gt_imgs=labels,
            input_imgs=features)

        input_img = Image.fromarray(features[0][0].type(torch.uint8).detach().numpy()*255)
        pred_img = Image.fromarray(pred_img.detach().numpy())
        org_labels = Image.fromarray(org_labels.detach().numpy())

        input_img.save(args.save_path+f're_{name}_input.png')
        pred_img.save(args.save_path+f're_{name}_pred.png')
        org_labels.save(args.save_path+f're_{name}_org.png')


def vec2img(
    pred_imgs:torch.Tensor,
    gt_imgs:torch.Tensor,
    input_imgs:torch.Tensor
):

    one_img = pred_imgs[0].detach().cpu().numpy()
    one_label = gt_imgs[0].detach().cpu().numpy()

    one_img = np.argmax(one_img, axis=0)

    palette = np.zeros((one_img.shape[0],one_img.shape[0],3), dtype=np.uint8)
    org_palette = np.zeros((one_img.shape[0], one_img.shape[0], 3), dtype=np.uint8)

    for i in range(pred_imgs.shape[-1]):
        for j in range(pred_imgs.shape[-1]):
            idx = one_img[j,i]
            org_idx = one_label[j,i]
     
            palette[j,i] = COLOR_MAP[idx]
            org_palette[j,i] = COLOR_MAP[org_idx]

    images = torch.from_numpy(palette)
    labels = torch.from_numpy(org_palette)

    return images, labels
