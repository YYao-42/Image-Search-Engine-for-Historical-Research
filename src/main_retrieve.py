import argparse
import fnmatch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

from src.networks.imageretrievalnet import extract_vectors_PQ, init_network, extract_vectors, extr_selfmade_dataset
from src.datasets.testdataset import configdataset
from src.datasets.datahelpers import cid2filename
from src.utils.download import download_test
from src.utils.evaluate import compute_map_and_print
from src.utils.networks import load_network
from src.utils.general import get_data_root, htime, path_all_jpg, save_path_feature, tb_setup


# some conflicts between tensorflow and tensoboard 
# causing embeddings to not be saved properly in tb
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except:
    pass


datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k', 'revisitop1m','custom']

# test options
parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Example')
parser.add_argument('--network', '-n', metavar='NETWORK', default='resnet101-solar-best.pth', 
                    help="network to be evaluated. " )
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='roxford5k',
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
parser.add_argument('--deep-quantization', '-dq', dest='deep_quantization', action='store_true',
                    help='model with deep quantization (supervised PQ)')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")

# parse the arguments
args = parser.parse_args()

def main():
    args = parser.parse_args()

    # check if there are unknown datasets
    for dataset in args.datasets.split(','):
        if dataset not in datasets_names:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

    # check if test dataset are downloaded
    # and download if they are not
    # download_test(get_data_root())

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network
    net = load_network(network_name=args.network)
    net.mode = 'test'

    print(">>>> loaded network: ")
    print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))
    if len(ms)>1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening'] and not net.meta['deep_quantization']:
        msp = net.pool.p.item()
        print(">> Set-up multiscale:")
        print(">>>> ms: {}".format(ms))
        print(">>>> msp: {}".format(msp))
    else:
        msp = 1

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

    ############# RETRIEVE ###################

    # evaluate on test datasets
    datasets = args.datasets.split(',')
    for dataset in datasets:
        # summary_ranks  = tb_setup(os.path.join('outputs/ranks/', dataset, args.network))
        # summary_embeddings = tb_setup(os.path.join('outputs/embeddings/', dataset, args.network))
        start = time.time()


        # prepare config structure for the test dataset
        cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
        images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
        images_r_path = [os.path.relpath(path, get_data_root()) for path in images]
        # images_r_path = [cfg['im_fname'](cfg, i).split('/')[-1] for i in range(cfg['n'])]
        qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
        qimages_r_path = [os.path.relpath(path, get_data_root()) for path in qimages]
        # qimages_r_path = [cfg['qim_fname'](cfg, i).split('/')[-1] for i in range(cfg['nq'])]
        try:
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        except:
            bbxs = None  # for holidaysmanrot and copydays
        
        print('>> {}: Extracting...'.format(dataset))

        # extract database and query vectors
        print('>> {}: database images...'.format(dataset))
        if args.deep_quantization:
            quantized_vecs, CW_idx, Codewords, vecs = extract_vectors_PQ(net, images, args.image_size, transform)
            CW_idx = CW_idx.numpy()
            CW_idx = CW_idx.astype(int)
            Codewords = Codewords.numpy()
            vecs = vecs.numpy()
        else:
            vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, mode='test')
            vecs = vecs.numpy()

        print('>> {}: query images...'.format(dataset))
        if args.deep_quantization:
            _, _, _, qvecs = extract_vectors_PQ(net, qimages, args.image_size, transform)
            qvecs = qvecs.numpy()
        else:
            qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, mode='test')
            qvecs = qvecs.numpy()

        print('>> {}: Evaluating...'.format(dataset))

        save_path_feature(dataset + '_database', vecs, images_r_path)
        save_path_feature(dataset + '_query', qvecs, qimages_r_path)
        
        # Save paths abd features
        if args.deep_quantization:
            # Codewords = net.C
            # Codewords = Codewords.cpu().detach().numpy()
            file_CW_idx = 'outputs/' + dataset + '_CW_idx.npy'
            file_Codewords = 'outputs/' + dataset + '_Codewords.npy'
            np.save(file_CW_idx, CW_idx)
            np.save(file_Codewords, Codewords)


        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset, ranks, cfg['gnd'])

        print('')

        # print('')
        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))

    print('>> selfmadedataset: Extracting...')
    # Extract features of selfmade datasets
    # extr_selfmade_dataset(net, 'custom/database', args.image_size, transform, ms, msp)
    # start = time.time()
    # extr_selfmade_dataset(net, 'GLM/test', args.image_size, transform, ms, msp)
    # extract_time= time.time() - start
    # extract_time_per = extract_time/1129
    # print('>> GLM : extracted time per image {}'.format(extract_time_per))
    # extr_selfmade_dataset(net, 'GLM/index', args.image_size, transform, ms, msp)

if __name__ == '__main__':
    main()
