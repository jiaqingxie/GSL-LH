"""
Script for running finetuning on glue tasks.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import argparse
import logging
import os
import csv
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.utils.prune as prune
from transformers import (
    AdamW, AutoConfig, AutoTokenizer
)






def positive_mask_scores(model):
    # transform mask_scores to a positive value and then it used for pruning.


    module =  model.gc1.weight_mask
    module.data = torch.sigmoid(module.data)
    module =  model.gc2.weight_mask 
    module.data = torch.sigmoid(module.data)


def pruning_mask_score(model, px):
    """
    Pruning mask score;
    mask score will be translated to mask through

    :param model:
    :param px: sparsity of mask
    :return:
    """

    parameters_to_prune = []

    parameters_to_prune.append((model.gc1, 'weight_mask'))
    parameters_to_prune.append((model.gc2, 'weight_mask'))



    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def draw_ticket_mask(model, sparsity):
    # draw masks of the robust ticket with a certain sparsity
    positive_mask_scores(model)
    pruning_mask_score(model, px=sparsity)
    mask_scores_mask_dict = {}
    model_dict = model.state_dict()
    for key in model_dict.keys():
        if 'mask_mask' in key:
            mask_scores_mask_dict[key] = model_dict[key]

    return mask_scores_mask_dict


def init_mask_score(model, ticket_mask):
        # query
    module = model.gc1.weight_mask

    module_mask = 'gc1.weight_mask_mask'
    mask = ticket_mask[module_mask]
    module.data = mask
        # key

    module = model.gc2.weight_mask
    module_mask = 'gc2.weight_mask_mask'
    mask = ticket_mask[module_mask]
    module.data = mask

