import argparse
import fnmatch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import time
import warnings

import numpy as np

import torch
from torchvision import transforms

from src.networks.imageretrievalnet import init_network, extract_vectors, extr_selfmade_dataset
from src.datasets.testdataset import configdataset
from src.utils.general import get_data_root
from src.utils.networks import load_network
from src.utils.nnsearch import *

# test options
parser = argparse.ArgumentParser(description='Historical Image Retrieval')
parser.add_argument('--network', '-n', metavar='NETWORK', default='resnet101-solar-best.pth', 
                    help="network to be evaluated. ")
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='roxford5k,rparis6k',
                    help="comma separated list of test datasets (default: 'roxford5k,rparis6k')")
parser.add_argument('--image-size', '-imsize', dest='image_size', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1, 2**(1/2), 1/2**(1/2)]',
                    help="use multiscale vectors for testing, " +
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--soa', action='store_true',
                    help='use soa blocks')
parser.add_argument('--soa-layers', type=str, default='45',
                    help='config soa blocks for second-order attention')
parser.add_argument('--K-nearest-neighbour', '-K', default=30, type=int, metavar='K',
                    help="retreive top-K results (default: 30)")
parser.add_argument('--matching_method', '-mm', default='L2', help="select matching methods: L2, PQ, ANNOY, HNSW, PQ_HNSW")
parser.add_argument('--ifgenerate', '-gen', dest='ifgenerate', action='store_true',
                    help='Include --ifgenerate if the trees/graphs/distance tables have not been generated and saved')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")

warnings.filterwarnings("ignore", category=UserWarning)


args = parser.parse_args()

# setting up the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# loading network
net = load_network(network_name=args.network)
net.mode = 'test'

print(">>>> loaded network: ")
print(net.meta_repr())

# setting up the multi-scale parameters
ms = list(eval(args.multiscale))

print(">>>> Evaluating scales: {}".format(ms))

# moving network to gpu and eval mode
net.cuda()
net.eval()

# set up the transform
normalize = transforms.Normalize(
    mean=net.meta['mean'],
    std=net.meta['std']
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# Extract features of different datasets
# The features will be saved automatically
# Pay attention to the paths
datasets = args.datasets.split(',')
for dataset in datasets:
    print('>> {}: Extracting...'.format(dataset))
    extr_selfmade_dataset(net, dataset, args.image_size, transform, ms)

dim_vec = 2048
vecs = np.empty((dim_vec, 0))
img_paths = []
for dataset in datasets:
    file_path_feature = 'outputs/features/' + dataset + '_path_feature.pkl'
    with open(file_path_feature, 'rb') as pickle_file:
        path_feature = pickle.load(pickle_file)
    vecs = np.concatenate([vecs, path_feature['feature']], axis=1)
    images = ['/static/' + i for i in path_feature['path']]
    img_paths = img_paths + images

# During the offline procedure, qvec doesn't matter. It can be anything since the construction of tree, graph, etc does not
# depend on qvec. Similar for K.
qvec = np.zeros((dim_vec, 1))
K = args.K_nearest_neighbour

# Note: parameters like N_books, n_bits_perbook, n_trees, etc will significantly influence the retrieval performance and efficiency
# Usually larger values lead to better performance but slower retrieval time
# If you want to change default values, don't forget to update the changes in online.py 
if args.matching_method == 'L2':
    match_idx, _ = matching_L2(K, vecs.T, qvec.T)
elif args.matching_method == 'PQ':
    match_idx, _ = matching_Nano_PQ(K, vecs.T, qvec.T, dataset='database', N_books=16, n_bits_perbook=13, ifgenerate=args.ifgenerate)
elif args.matching_method == 'ANNOY':
    match_idx, _ = matching_ANNOY(K, vecs.T, qvec.T, 'euclidean', dataset='database', n_trees=100, ifgenerate=args.ifgenerate)
elif args.matching_method == 'HNSW':
    match_idx, _ = matching_HNSW(K, vecs.T, qvec.T, dataset='database', m=16, ef=100, ifgenerate=args.ifgenerate)
elif args.matching_method == 'PQ_HNSW':
    match_idx, _ = matching_HNSW_NanoPQ(K, vecs.T, qvec.T, dataset='database', N_books=16, N_words=2**13, m=16, ef=100, ifgenerate=args.ifgenerate)
else:
    print('Invalid method')