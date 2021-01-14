############################################################
#   File: engine.py                                        #
#   Created: 2019-11-20 15:02:13                           #
#   Author : wvinzh                                        #
#   Email : wvinzh@qq.com                                  #
#   ------------------------------------------             #
#   Description:engine.py                                  #
#   Copyright@2019 wvinzh, HUST                            #
############################################################
import time
from utils import AverageMeter, compute_score_with_logits
import torch
import numpy as np
from utils.utils import cosine_annealing


class Engine():
    def __init__(self,):
        pass

    def train(self, state, epoch):
        # Initialize logger
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        config = state['config']
        print_freq = config.print_freq
        model = state['model']
        criterion = state['criterion']
        optimizer = state['optimizer']
        train_loader = state['train_loader']
        model.train()
        end = time.time()
        N = len(state['train_loader']) * config.epochs
        for i, (img, label, _) in enumerate(train_loader):
            if config.scheduler == 'cosine_annealing':
                optimizer.param_groups[0]['lr'] = cosine_annealing(state['step'], N, 20, state['lr'])
            f_loss = 0
            # measure data loading time
            data_time.update(time.time() - end)

            target = label.to(state['device'])
            input = img.to(state['device'])
            # Forward
            logits, _, _ = model(input)

            f_logits = logits

            org_loss = criterion(logits, target)

            f_loss = criterion(f_logits, target)

            loss = (org_loss + f_loss) / 2

            # measure accuracy and record loss
            score = compute_score_with_logits(logits, target).sum()
            losses.update(loss.item(), input.size(0))
            top1.update(score, input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1))
                print("Original loss, Fusion loss, Total loss", org_loss.item(), f_loss.item(),
                      loss.item())
            state['step'] += 1
        return top1.avg, losses.avg
    
    def validate(self, state):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        config = state['config']
        print_freq = config.print_freq
        model = state['model']
        val_loader = state['val_loader']
        criterion = state['criterion']
        confusion_matrix = np.zeros((8, 8))
        # switch to evaluate mode
        f_loss = 0
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (input, target, _) in enumerate(val_loader):
                target = target.cuda()
                input = input.cuda()
                # forward
                logits, _, _ = model(input)

                org_loss = criterion(logits, target)

                loss = (org_loss + f_loss) / 2

                score = compute_score_with_logits(logits, target).sum()
                losses.update(loss.item(), input.size(0))
                top1.update(score, input.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        .format(
                            i, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1))

            print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        print(confusion_matrix)
        return top1.avg, losses.avg

