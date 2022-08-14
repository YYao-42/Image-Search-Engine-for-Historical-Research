import argparse
import fnmatch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import time
import warnings

import numpy as np

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

from src.networks.imageretrievalnet import init_network, extract_vectors
from src.datasets.testdataset import configdataset
from src.utils.download import download_test
from src.utils.evaluate import compute_map_and_print
from src.utils.general import get_data_root, htime, save_path_feature, load_path_features
from src.utils.networks import load_network
from src.utils.Reranking import *


datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k', 'revisitop1m']

# test options
parser = argparse.ArgumentParser(description='Historical Image Retrieval')
parser.add_argument('--network', '-n', metavar='NETWORK', default='resnet101-solar-best.pth', 
                    help="network to be evaluated. ")
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='roxford5k,rparis6k',
                    help="comma separated list of test datasets: " +
                        " | ".join(datasets_names) +
                        " (default: 'roxford5k,rparis6k')")
parser.add_argument('--image-size', '-imsize', dest='image_size', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1, 2**(1/2), 1/2**(1/2)]',
                    help="use multiscale vectors for testing, " +
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--soa', action='store_true',
                    help='use soa blocks')
parser.add_argument('--soa-layers', type=str, default='45',
                    help='config soa blocks for second-order attention')
parser.add_argument('--include1m', '-1m', dest='include1m', action='store_true',
                    help='Whether include 1 million distractors')
parser.add_argument('--ifextracted', '-extracted', dest='ifextracted', action='store_true',
                    help='Whether the features have been extracted')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    args = parser.parse_args()

    # check if there are unknown datasets
    for dataset in args.datasets.split(','):
        if dataset not in datasets_names:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

    # check if test dataset are downloaded
    # and download if they are not
    download_test(get_data_root())

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

    # evaluate on test datasets
    datasets = args.datasets.split(',')
    for dataset in datasets:
        start = time.time()

        # prepare config structure for the test dataset
        cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
        images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
        images_r_path = [os.path.relpath(path, get_data_root()) for path in images]
        qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
        qimages_r_path = [os.path.relpath(path, get_data_root()) for path in qimages]
        try:
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        except:
            bbxs = None  # for holidaysmanrot and copydays

        if not args.ifextracted:
            # extract database and query vectors

            print('>> {}: Extracting...'.format(dataset))

            print('>> {}: database images...'.format(dataset))
            vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, mode='test')
            vecs = vecs.numpy()

            print('>> {}: query images...'.format(dataset))
            qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, mode='test')
            qvecs = qvecs.numpy()

            save_path_feature(dataset + '_db', vecs, images_r_path)
            save_path_feature(dataset + '_query', qvecs, qimages_r_path)
        else:
            vecs, _ = load_path_features(dataset + '_db')
            qvecs, _ = load_path_features(dataset + '_query')

        if args.include1m: # whether to include 1 million distractors
            vecs_1m = torch.load(args.network + '_vecs_' + 'revisitop1m' + '.pt')
            vecs_1m = vecs_1m.numpy()
            vecs = np.concatenate([vecs, vecs_1m], axis=1)

        print('>> {}: Evaluating...'.format(dataset))

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)

        # re-rank
        isExist = os.path.exists('src/diffusion/tmp/'+ dataset)
        if not isExist:
            os.makedirs('src/diffusion/tmp/'+ dataset)
        cache_dir = 'src/diffusion/tmp/' + dataset
        gnd_path2 = 'data/test/' + dataset + '/gnd_' + dataset + '.pkl'
        QGE(ranks, qvecs, vecs, dataset, gnd, cache_dir, gnd_path2, AQE=True)

        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


if __name__ == '__main__':
    main()
