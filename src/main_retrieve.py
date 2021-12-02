"""
Image Search Engine for Historical Research: A Prototype
Run this file to generate feature vectors of the images in the database,
and retrieve relevant images of given queries
"""

import time
import argparse

import torch
import numpy as np

import torchvision
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

from extractor import *
from nnsearch import *
from utils import *

# Instantiate the parser
parser = argparse.ArgumentParser()

# product quantization setting
parser.add_argument('--sup_PQ', type=bool, default=True,
                    help='whether using supervised product quantization')

# parse the arguments
args = parser.parse_args()

def main():
    cuda = torch.cuda.is_available()
    sup_PQ = args.sup_PQ
    if sup_PQ:
        net = TripletNet_PQ(resnet18(), Soft_PQ())
    else:
        net = TripletNet(resnet18())

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='../data/CIFAR', train=True,
                                                 download=True, transform=transform)
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                                       **kwargs)

    if sup_PQ:
        _, quantized_features, CW_idx, labels = get_feature_PQ(train_loader, net, cuda)
        # save_feature_PQ(quantized_features, labels, CW_idx)
    else:
        embedded_features, labels = get_feature(train_loader, net, cuda)
        # save_feature(embedded_features, labels)


    #############
    # RETRIEVAL
    #############
    test_dataset = torchvision.datasets.CIFAR10(root='../data/CIFAR', train=False,
                                                download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              **kwargs)
    N_query = 100
    K = 100
    t1_extra = time.time()
    if sup_PQ:
        embedded_features_test, _, _, labels_test = get_feature_PQ(test_loader, net, cuda)
    else:
        embedded_features_test, labels_test = get_feature(test_loader, net, cuda)
    t2_extra = time.time()
    t_extra = (t2_extra-t1_extra)/10000

    if sup_PQ:
        Codewords = net.C
        Codewords = Codewords.cpu().detach().numpy()
        N_books = 128
        hard_quantized_features = hard_quantization(Codewords, CW_idx, N_books)
        match_idx, time_per_query = matching_PQ_Net(K, Codewords, embedded_features_test[:N_query, :], N_books, CW_idx)
    else:
        # match_idx, time_per_query = matching_L2(K, embedded_features, embedded_features_test[:N_query, :])
        match_idx, time_per_query = matching_PQ_faiss(K, embedded_features, embedded_features_test[:N_query, :])

    mAP = cal_mAP(match_idx, labels, labels_test)
    print("extracting time per query: ", t_extra)
    print("matching time per query: ", time_per_query)
    print("Mean Average Precision: ", mAP)

if __name__ == '__main__':
    main()