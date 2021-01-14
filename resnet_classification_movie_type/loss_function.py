import torch
import torch.nn as nn


def boosted_loss(pred, a):
    alpha = 1e-9
    pred = torch.sigmoid(pred).squeeze(1)
    # pred = a*pred + (1-a) * (1-pred)
    # alpha = a*alpha + (1-a) * alpha
    # pred = torch.cat([pred.unsqueeze(1), alpha.unsqueeze(1)], 1)
    # pred = pred.max(1)[0]
    loss = -torch.log(pred + alpha) * a + torch.sqrt(alpha - torch.log(1-pred + alpha)) * (1 - a)
    # loss = -loss
    return loss.mean()


def BCE(pred, a):
    # pred = torch.sigmoid(pred)
    # a_max = pred.clamp(min=0)
    a_max = pred.max()  # [0].unsqueeze(1)
    # neg_pred = -pred.abs()
    loss = (torch.exp(pred - a_max).log() - (torch.exp(pred-a_max) + torch.exp(-a_max)).log()) * a - \
           ((torch.exp(pred - a_max) + torch.exp(-a_max)).log() + a_max) * (1-a)
    # loss1 = a_max - pred*a + (1 + torch.exp(neg_pred)).log()
    # loss = a * (1 + neg_pred.exp()).log() + \
    #        (a_max - pred * a + (1 + neg_pred.exp()).log() - (1 + neg_pred.exp()).log() * a)
    # weight_loss = loss * mask_weight
    # loss = loss.sum() + weight_loss.sum()
    return -loss.sum()
