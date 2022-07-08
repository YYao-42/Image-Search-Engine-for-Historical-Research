'''
1 libs import
'''
import os
import os.path
import numpy as np
import time
import argparse
import parser
import pickle
from tqdm import tqdm
import torch
import gc
# from src.main_retrieve import load_path_features
from utils.nnsearch import *
from datasets.testdataset import configdataset
from utils.evaluate import compute_map_and_print
from utils.evaluate import mAP_GLM
from utils.general import get_data_root
from utils.dataset import Dataset
from utils.diffusion import Diffusion
from utils.knn import KNN
from utils.evaluate2 import compute_map_and_print2
from sklearn import preprocessing
from utils.Reranking import *
from utils.general import load_path_features

'''
2 args setting
'''
parser = argparse.ArgumentParser(description='test: google landmark')
parser.add_argument('--matching_method', '-mm', default='L2', help="select matching methods: L2, PQ, ANNOY, HNSW, PQ_HNSW")
args = parser.parse_args()

'''
3 normal datasets
'''
# datasets = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k', 'revisitop1m']
datasets = ['roxford5k']

for dataset in datasets:
    '''
    We use the following two lines to read features (ro/rp).
    '''
    file_vecs = 'outputs/' + dataset + '_vecs.npy'
    file_qvecs = 'outputs/' + dataset + '_qvecs.npy'
    '''
    We use the following two lines to read features (large datasets).
    '''
    # file_vecs = 'outputs/solar/' + dataset + '_vecs.npy'
    # file_qvecs = 'outputs/solar/' + dataset + '_qvecs.npy'
    
    vecs = np.load(file_vecs)
    '''
    The following code is used for testing 1m datasets
    '''
    # vecs_1m = torch.load('outputs/' + 'resnet101-solar-best.pth' + '_vecs_' + 'revisitop1m' + '.pt')
    # vecs_1m = vecs_1m.numpy()
    # print(vecs_1m.shape) # roxford:(2048, 1001001)
    # vecs = np.concatenate((vecs, vecs_1m), axis=1)
    # print(vecs.shape) # roxford:(2048, 1005994)

    qvecs = np.load(file_qvecs)

    # flickr100k, _ = load_path_features('flickr100k')
    # vecs = np.concatenate((vecs, flickr100k), axis=1)
    n_database = vecs.shape[1]
    K = 4000 # This will also influence speed and accuracy

    '''
    For accuracy, we can use L2; for speed, we can use ANNOY. 
    '''
    match_idx, time_per_query = matching_L2(K, vecs.T, qvecs.T)
    # match_idx, time_per_query = matching_ANNOY(K, vecs.T, qvecs.T, 'euclidean', 'roxford5k', ifgenerate=True)

    print('matching time per query: ', time_per_query)
    ranks = match_idx.T
    cfg = configdataset(dataset, os.path.join('/home/qzhang7/data', 'test')) # This address is finding cfg. Cfg is used for testing results.
    images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
    qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
    print('------------------------------------------------------')
    print('mAP:')
    compute_map_and_print2(dataset, ranks, cfg['gnd'])
    print('------------------------------------------------------')

    query_num = qvecs.shape[1]
    sift_q_main_path = '/home/qzhang7/SIFT_features/' + dataset + '/query/' # The address of the offline part
    sift_g_main_path = '/home/qzhang7/SIFT_features/' + dataset + '/gallery/' # The address of the offline part
    gnd = cfg['gnd']
    cache_dir = '/users/qzhang7/src_qz/4_src-base/diffusion/tmp/' + dataset
    gnd_path2 = '/home/qzhang7/data/test/' + dataset + '/gnd_' + dataset + '.pkl'
    loftr_weight_path = "/users/qzhang7/src_qz/4_src-base/utils/weights/outdoor_ds.ckpt"
    K = 100
    AQE = True
    RW = True # For datasets (small than 120,000), turn it on (True), else, turn it off (False).

    # Query and Gallery Enhancement
    QGE(ranks, qvecs, vecs, dataset, gnd, query_num, cache_dir, gnd_path2, RW, AQE)

    # Query Expansion 2 (average)
    # average_query_expansion(qvecs, vecs, K, dataset, gnd)

    # Database Augmentation
    # database_augmentation(qvecs, vecs, K, dataset, gnd)

    # kr_reranking
    # kr_reranking(qvecs, vecs, dataset, gnd)

    # SAHA
    # sift_online(query_num, qimages, sift_q_main_path, images, sift_g_main_path, ranks, dataset, gnd)

    # LoFTR
    # loftr(loftr_weight_path, query_num, qimages, ranks, images, dataset, gnd)

    # sift ransac
    # ransac_sift(query_num, qimages, images, ranks, dataset, gnd)
