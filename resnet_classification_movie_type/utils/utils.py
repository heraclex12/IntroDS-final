
import os
import random
import numpy as np
import torch
import shutil
import logging
import _pickle as cPickle
from math import *

def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    set_seed(worker_id)
    # np.random.seed()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
        return old_lr


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def save_checkpoint(state, is_best, path='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)
    torch.save(state, full_path)
    if is_best:
        shutil.copyfile(full_path, os.path.join(path, 'model_best.pth.tar'))
        print("Save best model at %s==" %
              os.path.join(path, 'model_best.pth.tar'))


def cosine_annealing(step, n_iters, n_cycles, lrate_max):
    iter_per_cycle = n_iters / n_cycles
    cos_inner = (pi * (step % iter_per_cycle)) / (iter_per_cycle)
    lr = lrate_max / 2 * (cos(cos_inner) + 1)
    return lr
