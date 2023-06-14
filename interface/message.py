# -- coding : utf-8 -*-

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
Last modified : 2023.05.23
Code version : 1.0
"""


def print_current_epoch(
    epoch: int,
    tag:str='train'
):  
    if tag == 'train':
        print(f'\n[INFO] current epoch is : {epoch}')
    else:
        raise ValueError

def print_progress(
    bar:object,
    batch_iter:int,
    loss:float
):
    bar.update(batch_iter, values=[("loss: ", loss)])

def print_val(
    loss:float,
    val_loader_len:int
):
    print(f'\n       validation step | loss {loss/val_loader_len}')
    