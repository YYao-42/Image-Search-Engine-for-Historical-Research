"""
Image Search Engine for Historical Research: A Prototype
This file contains classes/functions related to data compression and nearest neighbor search
"""

import time
import argparse

import torch
import numpy as np

import torchvision
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

from numpy import linalg as LA
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import faiss
from extractor import *

def matching_L2(K, embedded_features_train, embedded_features_test):
    t1 = time.time()
    num_train, feature_len = embedded_features_train.shape
    num_test, _ = embedded_features_test.shape
    idx = np.zeros((num_test, K), dtype=np.int64)
    # normalization
    eftrain_norm = np.linalg.norm(embedded_features_train, axis=1)
    eftrain_norm = np.expand_dims(eftrain_norm, axis=1)
    eftest_norm = np.linalg.norm(embedded_features_test, axis=1)
    eftest_norm = np.expand_dims(eftest_norm, axis=1)
    embedded_features_train = embedded_features_train / eftrain_norm
    embedded_features_test = embedded_features_test / eftest_norm
    for row in range(num_test):
        query = embedded_features_test[row, :]
        dist = np.linalg.norm(query-embedded_features_train, axis=1)
        idx[row, :] = np.argpartition(dist, K-1)[:K]
    t2 = time.time()
    time_per_query = (t2-t1)/num_test
    return idx, time_per_query


def matching_L2_faiss(K, embedded_features_train, embedded_features_test):
    t1 = time.time()
    num_train, feature_len = embedded_features_train.shape
    num_test, _ = embedded_features_test.shape
    # normalization
    eftrain_norm = np.linalg.norm(embedded_features_train, axis=1)
    eftrain_norm = np.expand_dims(eftrain_norm, axis=1)
    eftest_norm = np.linalg.norm(embedded_features_test, axis=1)
    eftest_norm = np.expand_dims(eftest_norm, axis=1)
    embedded_features_train = embedded_features_train / eftrain_norm
    embedded_features_test = embedded_features_test / eftest_norm

    index = faiss.IndexFlatL2(feature_len)  # build the index
    index.add(embedded_features_train)  # add vectors to the index
    _, idx = index.search(embedded_features_test, K)
    t2 = time.time()
    time_per_query = (t2-t1)/num_test
    return idx, time_per_query


def matching_PQ_faiss(K, embedded_features_train, embedded_features_test):
    t1 = time.time()
    num_train, feature_len = embedded_features_train.shape
    num_test, _ = embedded_features_test.shape
    # normalization
    eftrain_norm = np.linalg.norm(embedded_features_train, axis=1)
    eftrain_norm = np.expand_dims(eftrain_norm, axis=1)
    eftest_norm = np.linalg.norm(embedded_features_test, axis=1)
    eftest_norm = np.expand_dims(eftest_norm, axis=1)
    embedded_features_train = embedded_features_train / eftrain_norm
    embedded_features_test = embedded_features_test / eftest_norm

    res = faiss.StandardGpuResources()
    code_size = 32
    ncentroids = 64 # np.floor(np.sqrt(feature_len)).astype('int64')  # sqrt(l)
    coarse_quantizer = faiss.IndexFlatL2(feature_len)
    index = faiss.IndexIVFPQ(coarse_quantizer, feature_len, ncentroids, code_size, 8)
    index.nprobe = 5
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.train(embedded_features_train)
    gpu_index.add(embedded_features_train)
    _, idx = gpu_index.search(embedded_features_test, K)
    t2 = time.time()
    time_per_query = (t2-t1)/num_test
    return idx, time_per_query


def matching_PQ_Net(K, Codewords, Query, N_books, CW_idx):
    '''

    Args:
        K: nearest K neighbors
        Codewords: N_words * (N_books * L_word)
        Query: N_query * (N_books * L_word)
        N_books: number of codebooks N_words: number of codewords per codebook L_word: length of codewords
        CW_idx: N_train_images * N_books

    Returns:
        idx: matching index
        time_per_query: matching time per query

    '''
    t1 = time.time()
    N_words, dim = Codewords.shape
    N_query, _ = Query.shape
    L_word = int(dim/N_books)
    Query = T.from_numpy(Query)
    q = T.split(Query, L_word, 1)
    Codewords = T.from_numpy(Codewords)
    c = T.split(Codewords, L_word, 1)
    # Generate a distance table: N_query * N_words * N_books
    for i in range(N_books):
        if i == 0:
            dist_table = squared_distances(q[i], c[i])
            dist_table = T.unsqueeze(dist_table, 2)
        else:
            temp = squared_distances(q[i], c[i])
            temp = T.unsqueeze(temp, 2)
            dist_table = T.cat((dist_table, temp), dim=2)
    dist_table = dist_table.cpu().detach().numpy()
    idx = np.zeros((N_query, K), dtype=np.int64)
    for i in range(N_query):
        dtable_per_query = dist_table[i, :, :]
        d_query_to_train = np.sum(dtable_per_query[CW_idx, range(N_books)], axis=1)
        idx[i, :] = np.argpartition(d_query_to_train, K-1)[:K]
    t2 = time.time()
    time_per_query = (t2 - t1) / N_query
    return idx, time_per_query


def cal_mAP(idx, labels_train, labels_test):
    num_queries, K = idx.shape
    matched = np.zeros_like(idx, dtype=np.int8)
    for i in range(num_queries):
        count = 0
        for j in range(K):
            if labels_test[i] == labels_train[idx[i, j]]:
                count += 1
                matched[i, j] = count
    N_truth = np.max(matched, axis=1, keepdims=True)+1e-16
    AP = np.sum(matched/(np.array(range(K))+1)/N_truth, axis=1)
    mAP = AP.mean()
    return mAP
