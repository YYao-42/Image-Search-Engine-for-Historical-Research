"""
Image Search Engine for Historical Research: A Prototype
Run this file to generate feature vectors of the images in the database,
and retrieve relevant images of given queries
"""

import argparse
import os
import time
import pickle
import pdb
from tqdm import tqdm
import xlrd
from copy import deepcopy

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms
import torch.nn as nn
import torch.utils.data
import torch.quantization
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
from sklearn import preprocessing

from imageretrievalnet import init_network, extract_vectors
from datasets.datahelpers import cid2filename
from datasets.testdataset import configdataset
from utils.download import download_train, download_test
from layers.whiten import whitenlearn, whitenapply
from utils.evaluate import compute_map_and_print
from utils.general import get_data_root, htime
from extractor import *
from utils.utils import *
from utils.nnsearch import *
from utils.adalam import AdalamFilter
from utils.src.utils.plotting import make_matching_figure
from utils.src.loftr import LoFTR, default_cfg
from utils.adalam import AdalamFilter
from utils.dataset import Dataset
from utils.diffusion import Diffusion
from utils.knn import KNN
from utils.evaluate2 import compute_map_and_print2

PRETRAINED = {
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'rSfM120k-tl-resnet101-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    }

datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

# Instantiate the parser
parser = argparse.ArgumentParser(description='Historical Image Retrieval Testing')

# product quantization setting，这个yyy里面没有
parser.add_argument('--sup_PQ', type=bool, default=False,
                    help='whether using supervised product quantization')
# network setting
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-path', '-npath', metavar='NETWORK',
                    help="pretrained network or network path (destination where network is saved)")
group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK',
                    help="off-the-shelf network, in the format 'ARCHITECTURE-POOLING' or 'ARCHITECTURE-POOLING-{reg-lwhiten-whiten}'," +
                        " examples: 'resnet101-gem' | 'resnet101-gem-reg' | 'resnet101-gem-whiten' | 'resnet101-gem-lwhiten' | 'resnet101-gem-reg-whiten'")

# test options
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='oxford5k,paris6k',
                    help="comma separated list of test datasets: " +
                        " | ".join(datasets_names) +
                        " (default: 'oxford5k,paris6k')")
parser.add_argument('--image-size', '-imsize', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1]',
                    help="use multiscale vectors for testing, " +
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--whitening', '-w', metavar='WHITENING', default=None, choices=whitening_names,
                    help="dataset used to learn whitening for testing: " +
                        " | ".join(whitening_names) +
                        " (default: None)")
# parser.add_argument('--cache_dir', type=str, default='./cache', help="""Directory to cache""")
# parser.add_argument('--dataset_name', type=str, required=True, help="""Name of the dataset""")
# parser.add_argument('--query_path', type=str, required=True, help="""Path to query features""")
# parser.add_argument('--gallery_path', type=str, required=True, help="""Path to gallery features""")
# parser.add_argument('--gnd_path', type=str, help="""Path to ground-truth""")
# parser.add_argument('-n', '--truncation_size', type=int, default=1000, 
#                     help="""Number of images in the truncated gallery""")

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")

# parse the arguments
args = parser.parse_args()
args.kq, args.kd = 10, 50
args.cache_dir = './diffusion/tmp/roxford5k'
# args.query_path = './diffusion/data/query/qvecs.npy'
# args.gallery_path = './diffusion/data/gallery/vecs.npy'
args.gnd_path = './diffusion/data/gnd_roxford5k.pkl'
args.dataset_name = 'roxford5k'
args.truncation_size = 1000

def convert_kpts(cv2_kpts):
    keypoints = np.array([(x.pt[0], x.pt[1]) for x in cv2_kpts ]).reshape(-1, 2)
    scales = np.array([12.0* x.size for x in cv2_kpts ]).reshape(-1, 1)
    angles = np.array([x.angle for x in cv2_kpts ]).reshape(-1, 1)
    responses = np.array([x.response for x in cv2_kpts]).reshape(-1, 1)
    return keypoints, scales, angles, responses

def main():
    # check if there are unknown datasets
    for dataset in args.datasets.split(','):
        if dataset not in datasets_names:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

    # check if test dataset are downloaded
    # and download if they are not
    # download_train(get_data_root())
    download_test(get_data_root())

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network from path
    if args.network_path is not None:

        print(">> Loading network:\n>>>> '{}'".format(args.network_path))
        if args.network_path in PRETRAINED:
            # pretrained networks (downloaded automatically)
            state = load_url(PRETRAINED[args.network_path], model_dir=os.path.join(get_data_root(), 'networks'))
        else:
            # fine-tuned network from path
            state = torch.load(args.network_path)

        # parsing net params from meta
        # architecture, pooling, mean, std required
        # the rest has default values, in case that is doesnt exist
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['local_whitening'] = state['meta'].get('local_whitening', False)
        net_params['regional'] = state['meta'].get('regional', False)
        net_params['whitening'] = state['meta'].get('whitening', False)
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False

        # load network
        cuda = torch.cuda.is_available()
        sup_PQ = args.sup_PQ
        if sup_PQ:
           net = TripletNet_PQ(resnet18(), Soft_PQ())
        else:
           net = init_network(net_params)
           net.load_state_dict(state['state_dict'])

        # if whitening is precomputed
        if 'Lw' in state['meta']:
            net.meta['Lw'] = state['meta']['Lw']

        print(">>>> loaded network: ")
        print(net.meta_repr())

    # loading offtheshelf network
    elif args.network_offtheshelf is not None:

        # parse off-the-shelf parameters
        offtheshelf = args.network_offtheshelf.split('-')
        net_params = {}
        net_params['architecture'] = offtheshelf[0]
        net_params['pooling'] = offtheshelf[1]
        net_params['local_whitening'] = 'lwhiten' in offtheshelf[2:]
        net_params['regional'] = 'reg' in offtheshelf[2:]
        net_params['whitening'] = 'whiten' in offtheshelf[2:]
        net_params['pretrained'] = True

        # load off-the-shelf network
        print(">> Loading off-the-shelf network:\n>>>> '{}'".format(args.network_offtheshelf))
        net = init_network(net_params)
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))
    if len(ms)>1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
        msp = net.pool.p.item()
        print(">> Set-up multiscale:")
        print(">>>> ms: {}".format(ms))
        print(">>>> msp: {}".format(msp))
    else:
        msp = 1

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

    # compute whitening
    if args.whitening is not None:
        start = time.time()

        if 'Lw' in net.meta and args.whitening in net.meta['Lw']:

            print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))

            if len(ms)>1:
                Lw = net.meta['Lw'][args.whitening]['ms']
            else:
                Lw = net.meta['Lw'][args.whitening]['ss']

        else:

            # if we evaluate networks from path we should save/load whitening
            # not to compute it every time
            if args.network_path is not None:
                whiten_fn = args.network_path + '_{}_whiten'.format(args.whitening)
                if len(ms) > 1:
                    whiten_fn += '_ms'
                whiten_fn += '.pth'
            else:
                whiten_fn = None

            if whiten_fn is not None and os.path.isfile(whiten_fn):
                print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
                Lw = torch.load(whiten_fn)

            else:
                print('>> {}: Learning whitening...'.format(args.whitening))

                # loading db
                db_root = os.path.join(get_data_root(), 'train', args.whitening)
                ims_root = os.path.join(db_root, 'ims')
                db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(args.whitening))
                with open(db_fn, 'rb') as f:
                    db = pickle.load(f)
                images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

                # extract whitening vectors
                print('>> {}: Extracting...'.format(args.whitening))
                wvecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)

                # learning whitening
                print('>> {}: Learning...'.format(args.whitening))
                wvecs = wvecs.numpy()
                m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
                Lw = {'m': m, 'P': P}

                # saving whitening if whiten_fn exists
                if whiten_fn is not None:
                    print('>> {}: Saving to {}...'.format(args.whitening, whiten_fn))
                    torch.save(Lw, whiten_fn)

        print('>> {}: elapsed time: {}'.format(args.whitening, htime(time.time()-start)))

    else:
        Lw = None


    #############
    # RETRIEVAL
    #############
    datasets = args.datasets.split(',')
    for dataset in datasets:
        start = time.time()
        print('>> {}: Extracting...'.format(dataset))

        # prepare config structure for the test dataset
        cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
        images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
        #全部图片的绝对路径，list，读取相对应位置的路径即可，带.jpg
        images_r_path = [cfg['im_fname'](cfg, i).split('\\')[-1] for i in range(cfg['n'])]
        #全部文件的相对路径，list，带.jpg
        qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
        print(qimages)
        qimages_r_path = [cfg['qim_fname'](cfg, i).split('\\')[-1] for i in range(cfg['nq'])]
        try:
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        except:
            bbxs = None  # for holidaysmanrot and copydays

        # extract database and query vectors
        print('>> {}: database images...'.format(dataset))
        vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
        print('>> {}: query images...'.format(dataset))
        qextract_start = time.time()
        qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, msp=msp)
        qextract_end = time.time()
        qextract_per_query = (qextract_end - qextract_start)/55
        print('>> {}: Evaluating...'.format(dataset))
        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

        if Lw is not None:
            # whiten the vectors
            qextract_start_w = time.time()
            vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
            qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])
            qextract_end_w = time.time()
            qextract_per_query = (qextract_end_w - qextract_start_w)/55+qextract_per_query
            # for search & rank
            vecs = vecs_lw
            qvecs = qvecs_lw

        # Save features        

        isExist = os.path.exists('outputs')
        if not isExist:
            os.makedirs('outputs')

        file_vecs = 'outputs/' + dataset + '_vecs.npy'
        file_qvecs = 'outputs/' + dataset + '_qvecs.npy'

        np.save(file_vecs, vecs)
        np.save(file_qvecs, qvecs)

        vecs = np.load(file_vecs)
        qvecs = np.load(file_qvecs)

        # search, rank, and print
        n_database = vecs.shape[1]
        K = n_database
        # K = 10
        match_idx, time_per_query = matching_L2(K, vecs.T, qvecs.T)
        # match_idx, time_per_query = matching_Nano_PQ(K, vecs.T, qvecs.T, 16, 8)
        # match_idx, time_per_query = matching_LSH_faiss(K, vecs.T, qvecs.T, 128)
        # match_idx, time_per_query = matching_ANNOY(K, vecs.T, qvecs.T, 'euclidean')
        # match_idx, time_per_query = matching_HNSW(K, vecs.T, qvecs.T)
        # embedded_code, Codewords, _ = Nano_PQ(vecs.T, 16, 256)
        # match_idx, time_per_query = matching_PQ_Net(K, Codewords, qvecs.T, 16, embedded_code)
        # match_idx, time_per_query = matching_HNSW_PQ(K, Codewords, qvecs.T, embedded_code)
        print('matching time per query: ', time_per_query)
        ranks = match_idx.T
        compute_map_and_print(dataset, ranks, cfg['gnd'])

        # Reranking
        # device = torch.device('cpu')
        # try:
        #      if torch.cuda.is_available():
        #          device = torch.device('cuda')
        #          print ("GPU mode")
        # except:
        #      print ('CPU mode')
        # 1 SIFT
        # 1.2 SIFT online

        # def bubbleSort(nums):
        #     for i in range(nums.shape[0]-1):
        #          for j in range(nums.shape[0]-i-1):
        #              if nums[j,1] <= nums[j+1,1]:
        #                  nums[j,1], nums[j+1,1] = nums[j+1,1], nums[j,1]
        #                  nums[j,0], nums[j+1,0] = nums[j+1,0], nums[j,0]
        #     return nums

        # matcher = AdalamFilter()
        # b = 60

        # for i in range(0,70):
        #      a = np.zeros(shape=(b, 2))
        #      ranks_query_index = int(i)
        #      query_ab_path = qimages[ranks_query_index]
        #      query_name_jpg = query_ab_path.split('\\')[9]
        #      query_name = str(query_name_jpg.split('.')[0])
        #      h1_path = '/home/qzhang7/SIFT_features/query/'\
        #                + query_name + '_h1' + '.npy'
        #      w1_path = '/home/qzhang7/SIFT_features/query/'\
        #                + query_name + '_w1' + '.npy'
        #      query_descriptors_path = '/home/qzhang7/SIFT_features/query/'\
        #                                + query_name + '_descriptors' + '.npy'
        #      query_kp_path = '/home/qzhang7/SIFT_features/query/'\
        #                      + query_name + '_kp' + '.npy'
        #      query_s_path = '/home/qzhang7/SIFT_features/query/'\
        #                      + query_name + '_s' + '.npy'
        #      query_a_path = '/home/qzhang7/SIFT_features/query/'\
        #                      + query_name + '_a' + '.npy'
        #      query_r_path = '/home/qzhang7/SIFT_features/query/'\
        #                      + query_name + '_r' + '.npy'
        #      h1 = np.load(h1_path)
        #      w1 = np.load(w1_path)
        #      descriptors_query = np.load(query_descriptors_path)
        #      kp_q = np.load(query_kp_path)
        #      s_q = np.load(query_s_path)
        #      a_q = np.load(query_a_path)
        #      r_q = np.load(query_r_path)

        #      for j in range(0,b):
        #         ranks_image_index = int(ranks[j,i]) #ranks中可能没有最外面的那一行一列，所以需要重调+1
        #         image_ab_path = images[ranks_image_index]
        #         image_name_jpg = image_ab_path.split('\\')[9]
        #         image_name = str(image_name_jpg.split('.')[0])
        #         h2_path = '/home/qzhang7/SIFT_features/gallery/' +\
        #                   image_name + '_h2' + '.npy'
        #         w2_path = '/home/qzhang7/SIFT_features/gallery/' +\
        #                   image_name + '_w2' + '.npy'
        #         image_descriptors_path = '/home/qzhang7/SIFT_features/gallery/'\
        #                                    + image_name + '_descriptors' + '.npy'
        #         image_kp_path = '/home/qzhang7/SIFT_features/gallery/'\
        #                           + image_name + '_kp' + '.npy'
        #         image_s_path = '/home/qzhang7/SIFT_features/gallery/'\
        #                           + image_name + '_s' + '.npy'
        #         image_a_path = '/home/qzhang7/SIFT_features/gallery/'\
        #                           + image_name + '_a' + '.npy'
        #         image_r_path = '/home/qzhang7/SIFT_features/gallery/'\
        #                           + image_name + '_r' + '.npy'
        #         h2 = np.load(h2_path)
        #         w2 = np.load(w2_path)
        #         descriptors_image = np.load(image_descriptors_path)
        #         kp_i = np.load(image_kp_path)
        #         s_i = np.load(image_s_path)
        #         a_i = np.load(image_a_path)
        #         r_i = np.load(image_r_path)

        #         idxs = matcher.match_and_filter(kp_q, kp_i,
        #                                 descriptors_query, descriptors_image,
        #                                 im1shape=(h1, w1),
        #                                 im2shape=(h2, w2),
        #                                 o1=a_q.reshape(-1),
        #                                 o2=a_i.reshape(-1),
        #                                 s1=s_q.reshape(-1),
        #                                 s2=s_i.reshape(-1)).detach().cpu().numpy()

        #         a[j, 0] = ranks_image_index
        #         a[j, 1] = len(idxs)

        #      a = bubbleSort(a)
        #      for p in range(0, b):
        #          ranks[p,i]=a[p,0]

        # 2 Diffusion
        truncation_size = 1000
        cache_dir = '/users/qzhang7/src_qz/4_src-base/diffusion/tmp/roxford5k'
        kd = 50
        kq = 10
        diffusion = Diffusion(vecs.T, cache_dir)
        offline = diffusion.get_offline_results(truncation_size, kd)

        time0 = time.time()
        print('[search] 1) k-NN search')
        sims, ids = diffusion.knn.search(qvecs.T, kq)
        sims = sims ** 3
        qr_num = ids.shape[0]

        print('[search] 2) linear combination')
        all_scores = np.empty((qr_num, truncation_size), dtype=np.float32)
        all_ranks = np.empty((qr_num, truncation_size), dtype=np.int)
        for i in tqdm(range(qr_num), desc='[search] query'):
            scores = sims[i] @ offline[ids[i]]
            parts = np.argpartition(-scores, truncation_size)[:truncation_size]
            ranks = np.argsort(-scores[parts])
            all_scores[i] = scores[parts][ranks]
            all_ranks[i] = parts[ranks]
        print('[search] search costs {:.2f}s'.format(time.time() - time0))
        ranks2 = all_ranks.T
        gnd_path2 = '/users/qzhang7/src_qz/data/test/roxford5k/gnd_roxford5k.pkl'
        gnd_name2 = os.path.splitext(os.path.basename(gnd_path2))[0]
        with open(gnd_path2, 'rb') as f:
            gnd2 = pickle.load(f)['gnd']
        compute_map_and_print2(gnd_name2.split("_")[-1], ranks2, gnd2)
        
        # 3 LoFTR
        # _default_cfg = deepcopy(default_cfg)
        # _default_cfg['coarse']['temp_bug_fix'] = False
        # matcher = LoFTR(config=_default_cfg)
        # matcher.load_state_dict(torch.load("/users/qzhang7/src_qz/4_src-base/utils/weights/outdoor_ds.ckpt")['state_dict'])

        # matcher = matcher.eval().cuda()

        # def bubbleSort(nums):
        #      for i in range(nums.shape[0]-1):
        #         for j in range(nums.shape[0]-i-1):
        #            if nums[j,1] <= nums[j+1,1]:
        #               nums[j,1], nums[j+1,1] = nums[j+1,1], nums[j,1]
        #               nums[j,0], nums[j+1,0] = nums[j+1,0], nums[j,0]
        #      return nums

        # b = 50

        # T1 = time.time()

        # for i in range(0,70): 
        #     a = np.zeros(shape=(b, 2))
        #     ranks_query_index = int(i)
        #     query_path = qimages[ranks_query_index]
        #     query_name_jpg = query_path.split('\\').[9]
        #     query_name = query_name_jpg.split('.')[0]
        #     query_image = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
        #     query_image = cv2.resize(query_image, (720, 480))
        #     query_image_np = torch.from_numpy(query_image)[None][None].cuda() / 255.

        #     for j in range(0, b):
        #         ranks_image_index = int(ranks[j, i])
        #         image_path = images[ranks_image_index]
        #         image_name_jpg = image_path.split('\\').[9]
        #         image_name = image_name_jpg.split('.')[0]
        #         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #         image = cv2.resize(image, (720, 480))
        #         image_np = torch.from_numpy(image)[None][None].cuda() / 255.

        #         batch = {'image0': query_image_np, 'image1': image_np}

        #         with torch.no_grad():
        #             matcher(batch)
        #             mkpts0 = batch['mkpts0_f'].cpu().numpy()
        #             mkpts1 = batch['mkpts1_f'].cpu().numpy()
        #             mconf = batch['mconf'].cpu().numpy()

        #         color = cm.jet(mconf)
        #         text = ['LoFTR', 'Matches1: {}'.format(len(mkpts0)), 'Matches2: {}'.format(len(mkpts1))]
        #         fig = make_matching_figure(query_image, image, mkpts0, mkpts1, color, text=text)
        #         make_matching_figure(query_image, image, mkpts0, mkpts1, color, mkpts0, mkpts1, text,\
        #             path="/home/qzhang7/LoFTR_results/roxford/"+ str(query_name) +\
        #                   '_' + str(image_name) + ".jpg")
        #         a[j, 0] = ranks_image_index
        #         a[j, 1] = len(mkpts0)

        #     a = bubbleSort(a)
        #     for p in range(0, b):
        #        ranks[p,i]=a[p,0]

        # T2 = time.time()
        # print('Time for LoFTR reranking per query:%s' % ((T2 - T1)/70))


        # Output ranked images
        rank_res = {}
        for i in range(len(qimages_r_path)):
            rank_res[qimages_r_path[i]] = [images_r_path[j] for j in match_idx[i,:]]
        file_rankres = 'outputs/' + dataset + '_ranking_result.pkl'
        a_file = open(file_rankres, "wb")
        pickle.dump(rank_res, a_file)
        a_file.close()

        # Visualization
        # Visualize the selected query image and its matching images
        gnd = cfg['gnd']
        K_show = 20
        idx_select = 1
        query_image = qimages[idx_select]
        matching_images = [images[j] for j in match_idx[idx_select, :K_show]]
        plt.close('all')
        plt.figure(figsize=(10, 4), dpi=80)
        ax = plt.subplot2grid((2, K_show), (0, 0))
        ax.axis('off')
        ax.set_title('Query')
        img = mpimg.imread(query_image)
        plt.imshow(img)
        for i in range(K_show):
            if dataset == 'oxford5k' or dataset == 'paris6k':
                if np.in1d(match_idx[idx_select, i], gnd[idx_select]['ok'])[0]:
                    plt.rcParams["axes.edgecolor"] = "green"
                else:
                    plt.rcParams["axes.edgecolor"] = "red"
            if dataset == 'roxford5k' or dataset == 'rparis6k':
                if np.in1d(match_idx[idx_select, i], gnd[idx_select]['easy'])[0]:
                    plt.rcParams["axes.edgecolor"] = "green"
                elif np.in1d(match_idx[idx_select, i], gnd[idx_select]['hard'])[0]:
                    plt.rcParams["axes.edgecolor"] = "blue"
                elif np.in1d(match_idx[idx_select, i], gnd[idx_select]['junk'])[0]:
                    plt.rcParams["axes.edgecolor"] = "red"
            plt.rcParams["axes.linewidth"] = 2.50
            ax = plt.subplot2grid((2, K_show), (1, i))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_title('Match #' + str(i + 1))
            img = mpimg.imread(matching_images[i])
            plt.imshow(img)
        plt.tight_layout(pad=2)
        file_vis_path = 'outputs/' + dataset + '_' + str(idx_select) + '_vis.png'
        plt.savefig(file_vis_path)

        print("extracting time per query : ", qextract_per_query)
        # print("retrieve time per query: ", retrieve_per_query)
        print('>> {}: whole elapsed time: {}'.format(dataset, htime(time.time()-start)))


if __name__ == '__main__':
    main()
