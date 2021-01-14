import os
import argparse


def getConfig():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--image-size', '-i', default=512, type=int,
                        metavar='N', help='image size (default: 512)')
    parser.add_argument('--input-size', '-cs', default=448, type=int,
                        metavar='N', help='the input size of the model (default: 448)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # optimizer config
    parser.add_argument('--optim', default='sgd', type=str,
                        help='the name of optimizer(adam,sgd)')
    parser.add_argument('--scheduler', default='plateau', type=str,
                        help='the name of scheduler(step,plateau)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')

    # model config
    parser.add_argument('--model-name', default='resnet101', type=str,
                        help='model name')

    # training config
    parser.add_argument('--use-gpu', action="store_true", default=True,
                        help='whether use gpu or not, default True')
    parser.add_argument('--multi-gpu', action="store_true", default=False,
                        help='whether use multiple gpus or not, default True')
    parser.add_argument('--gpu-ids', default='0',
                        help='gpu id list(eg: 0,1,2...)')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--print-freq', '-pf', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint-path', default='saved_log/resnet101', type=str, metavar='checkpoint_path',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=str, default='saved_log/resnet101')


    #Input
    parser.add_argument('--image', type=str, default='sample_test.jpg', help='Image input')
    args = parser.parse_args()

    return args
