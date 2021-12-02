"""
Image Search Engine for Historical Research: A Prototype
Run this file to train the end-to-end model
"""

import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
from torch.utils.data import random_split
from torchvision import transforms
from utils import train
from extractor import *
from datasets import TripletCifar

# Instantiate the parser
parser = argparse.ArgumentParser()

# directory
parser.add_argument('--ckptroot', type=str,
                    default="../checkpoint/checkpoint.pth.tar", help='path to checkpoint')

# hyperparameters settings
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum', type=float,
                    default=0.9, help='momentum factor')
parser.add_argument('--nesterov', type=bool, default=True,
                    help='enables Nesterov momentum')
parser.add_argument('--weight_decay', type=float,
                    default=1e-5, help='weight decay (L2 penalty)')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int,
                    default=128, help='batch size')
parser.add_argument('--start_epoch', type=int,
                    default=0, help='starting epoch')

# loss function settings
parser.add_argument('--g', type=float, default=1.0, help='gap parameter')
parser.add_argument('--p', type=int, default=2,
                    help='norm degree for pairwise distance - Euclidean Distance')

# training settings
parser.add_argument('--resume', type=bool, default=False,
                    help='whether re-training from ckpt')
parser.add_argument('--is_gpu', type=bool, default=True,
                    help='whether training using GPU')

# product quantization setting
parser.add_argument('--sup_PQ', type=bool, default=True,
                    help='whether using supervised product quantization')

# parse the arguments
args = parser.parse_args()


def main():
    """Main pipeline"""
    if args.sup_PQ:
        net = TripletNet_PQ(resnet18(), Soft_PQ())
    else:
        net = TripletNet(resnet18())

    # For training on GPU, we need to transfer net and data onto the GPU
    if args.is_gpu:
        print("==> Initialize CUDA support for TripletNet model ...")
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    # optimizer = torch.optim.SGD(net.parameters(),
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay,
    #                             nesterov=args.nesterov)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # resume training from the last time
    if args.resume:
        # Load checkpoint
        print('==> Resuming training from checkpoint ...')
        checkpoint = torch.load(args.ckptroot)
        args.start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        valid_loss_min_input = checkpoint['valid_loss_min']
        print("==> Loaded checkpoint '{}' (epoch {})".format(
            args.ckptroot, checkpoint['epoch']))
    else:
        # start over
        print('==> Building new TripletNet model ...')
        valid_loss_min_input = np.Inf

    criterion = nn.TripletMarginLoss(margin=args.g, p=args.p)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=10,
                                                           verbose=True)

    # Load triplet dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='../data/CIFAR', train=True,
                                                 download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root='../data/CIFAR', train=False,
                                                 download=True, transform=transform)

    # Returns triplets of images
    triplet_train_dataset = TripletCifar(train_dataset)
    triplet_val_dataset = TripletCifar(val_dataset)

    # Dataloaders for triplets
    batch_size = args.batch_size
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True,
                                                       **kwargs)
    triplet_val_loader = torch.utils.data.DataLoader(triplet_val_dataset, batch_size=batch_size, shuffle=True,
                                                       **kwargs)

    # train model
    train(net, args.sup_PQ, criterion, optimizer, scheduler, triplet_train_loader,
          triplet_val_loader, args.start_epoch, args.epochs, args.is_gpu, valid_loss_min_input)


if __name__ == '__main__':
    main()
