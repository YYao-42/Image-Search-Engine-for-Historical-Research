'''
1 import libs
'''
import os
import numpy as np
from tqdm import tqdm
import argparse
import time
import pickle
import copy
from copy import deepcopy
import gc

import matplotlib.pyplot as plt
import cv2
import torch
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *

from utils.src.utils.plotting import make_matching_figure
from utils.src.loftr import LoFTR, default_cfg
from utils.adalam import AdalamFilter
from utils.dataset import Dataset
from utils.diffusion import Diffusion
from utils.knn import KNN
from utils.nnsearch import *
from utils.evaluate import compute_map_and_print
from utils.evaluate2 import compute_map_and_print2

'''
2 SAHA online
This is the online part of SAHA. We need to run SAHAoffline.py first to obtain features of all images.
Then we can run the online part to implement SAHA re-ranking. We can also use SAHA as a single file.
But in this way, SAHA will be much slower.
'''
def sift_online(query_num, qimages, sift_q_main_path, images, sift_g_main_path, ranks, dataset, gnd):

    device = torch.device('cpu')
    try:
         if torch.cuda.is_available():
             device = torch.device('cuda:0')
             print ("GPU mode")
    except:
         print ('CPU mode')

    def bubbleSort(nums):
        for i in range(nums.shape[0]-1):
             for j in range(nums.shape[0]-i-1):
                 if nums[j,1] <= nums[j+1,1]:
                     nums[j,1], nums[j+1,1] = nums[j+1,1], nums[j,1]
                     nums[j,0], nums[j+1,0] = nums[j+1,0], nums[j,0]
        return nums

    matcher = AdalamFilter()
    b = 30 # we use top 30 preliminary results to re-rank
    T_sift_1 = time.time()
    for i in tqdm(range(0,query_num)):
         a = np.zeros(shape=(b, 2)) # build a list to save matching numbers
         ranks_query_index = int(i)
         query_ab_path = qimages[ranks_query_index] # finding image path
         query_name_jpg = query_ab_path.split('/')[7]
         query_name = str(query_name_jpg.split('.')[0]) # finding image names
         h1_path = sift_q_main_path + query_name + '_h1' + '.npy' # height
         w1_path = sift_q_main_path + query_name + '_w1' + '.npy' # weight
         query_descriptors_path = sift_q_main_path + query_name + '_descriptors' + '.npy'
         query_kp_path = sift_q_main_path + query_name + '_kp' + '.npy' # keypoints
         query_s_path = sift_q_main_path + query_name + '_s' + '.npy' # size
         query_a_path = sift_q_main_path + query_name + '_a' + '.npy' # angle
         # query_r_path = sift_q_main_path + query_name + '_r' + '.npy'
         h1 = np.load(h1_path) # new height
         w1 = np.load(w1_path) # new weight
         descriptors_query = np.load(query_descriptors_path) # descriptors of queries
         kp_q = np.load(query_kp_path) # keypoints of queries
         s_q = np.load(query_s_path) # size of queries
         a_q = np.load(query_a_path) # angle of queries
         # r_q = np.load(query_r_path)

         for j in range(0,b):
            ranks_image_index = int(ranks[j,i])
            image_ab_path = images[ranks_image_index]  
            image_name_jpg = image_ab_path.split('/')[7]
            image_name = str(image_name_jpg.split('.')[0])
            h2_path = sift_g_main_path + image_name + '_h2' + '.npy'
            w2_path = sift_g_main_path + image_name + '_w2' + '.npy'
            image_descriptors_path = sift_g_main_path + image_name + '_descriptors' + '.npy'
            image_kp_path = sift_g_main_path + image_name + '_kp' + '.npy'
            image_s_path = sift_g_main_path + image_name + '_s' + '.npy'
            image_a_path = sift_g_main_path + image_name + '_a' + '.npy'
            # image_r_path = sift_g_main_path + image_name + '_r' + '.npy'
            h2 = np.load(h2_path)
            w2 = np.load(w2_path)
            descriptors_image = np.load(image_descriptors_path)
            kp_i = np.load(image_kp_path)
            s_i = np.load(image_s_path)
            a_i = np.load(image_a_path)
            # r_i = np.load(image_r_path)

            idxs = matcher.match_and_filter(kp_q, kp_i,
                                    descriptors_query, descriptors_image,
                                    im1shape=(h1, w1),
                                    im2shape=(h2, w2),
                                    o1=a_q.reshape(-1),
                                    o2=a_i.reshape(-1),
                                    s1=s_q.reshape(-1),
                                    s2=s_i.reshape(-1)).detach().cpu().numpy()

            a[j, 0] = ranks_image_index
            a[j, 1] = len(idxs)

         a = bubbleSort(a)
         for p in range(0, b):
             ranks[p,i]=a[p,0]
    compute_map_and_print(dataset, ranks, gnd)
    T_sift_2 = time.time()
    print('Time for SIFT reranking per query:%s' % ((T_sift_2-T_sift_1)/query_num))

'''
3 LoFTR
This is LoFTR. LoFTR was an image matching method. We modified it and made it as a re-ranking method.
LoFTR can be a little more accurate than SAHA. However, it is impossible to implement the offline/online
design on it. Therefore it is slower than SAHA.
'''
def loftr(loftr_weight_path, query_num, qimages, ranks, images, dataset, gnd):
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = False
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load(loftr_weight_path)['state_dict'])
    matcher = matcher.eval().cuda()

    def bubbleSort(nums):
        for i in range(nums.shape[0] - 1):
            for j in range(nums.shape[0] - i - 1):
                if nums[j, 1] <= nums[j + 1, 1]:
                    nums[j, 1], nums[j + 1, 1] = nums[j + 1, 1], nums[j, 1]
                    nums[j, 0], nums[j + 1, 0] = nums[j + 1, 0], nums[j, 0]
        return nums

    b = 60 # we use top 60 preliminary results for re-ranking
    resolution = (720, 480)

    T_loftr_1 = time.time()

    for i2 in tqdm(range(0, query_num)):
        a = np.zeros(shape=(b, 2))
        ranks_query_index = int(i2)
        query_path = qimages[ranks_query_index]
        query_image = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
        query_image = cv2.resize(query_image, resolution)
        query_image_np = torch.from_numpy(query_image)[None][None].cuda() / 255.

        for j2 in range(0, b):
            ranks_image_index = int(ranks[j2, i2])
            image_path = images[ranks_image_index]
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, resolution)
            image_np = torch.from_numpy(image)[None][None].cuda() / 255.

            batch = {'image0': query_image_np, 'image1': image_np}

            with torch.no_grad():
                matcher(batch)
                mkpts0 = batch['mkpts0_f'].cpu().numpy()
                mkpts1 = batch['mkpts1_f'].cpu().numpy()
                mconf = batch['mconf'].cpu().numpy()

            # color = cm.jet(mconf)
            # text = ['LoFTR', 'Matches1: {}'.format(len(mkpts0)), 'Matches2: {}'.format(len(mkpts1))]
            # fig = make_matching_figure(query_image, image, mkpts0, mkpts1, color, text=text)
            # make_matching_figure(query_image, image, mkpts0, mkpts1, color, mkpts0, mkpts1, text,\
            #     path="/home/qzhang7/LoFTR_results/roxford/"+ str(query_name) +\
            #           '_' + str(image_name) + ".jpg")
            a[j2, 0] = ranks_image_index
            a[j2, 1] = len(mkpts0)

        a = bubbleSort(a)
        for p in range(0, b):
            ranks[p, i2] = a[p, 0]  

    compute_map_and_print(dataset, ranks, gnd)  

    T_loftr_2 = time.time()
    print('Time for LoFTR reranking per query:%s' % ((T_loftr_2 - T_loftr_1) / query_num))

'''
4 QGE
This is the method we used in our work. QGE is a global feature-based re-ranking method. It is very 
fast and accurate.
'''
def QGE(ranks, qvecs, vecs, dataset, gnd, query_num, cache_dir, gnd_path2, RW, AQE):
    def feature_enhancement(it_times, k, ranks, qvecs, vecs, w):
        for it_time in range(it_times):
            qe_weight = (np.arange(k, 0, -1) / k).reshape(1, k, 1) # build an array, [1, 1/2, ..., 1/k]
            ranks_top = ranks[:k, int(0): int(ranks.shape[1])]
            top_k_vecs = vecs[:, ranks_top]
            # If we have query images in databases, we can use the following line.
            qvecs_top = (top_k_vecs * (qe_weight ** w)).sum(axis=1)
            # If we don't have query images in databases, we can use the following line.
            # qvecs_top = (top_k_vecs * (qe_weight ** w)).sum(axis=1) + qvecs
            qvecs_top = qvecs_top / (np.linalg.norm(qvecs_top, ord=2, axis=0, keepdims=True) + 1e-6)
            qvecs_qe = qvecs_top
            scores_aqe = np.dot(vecs.T, qvecs_qe)
            ranks_aqe = np.argsort(-scores_aqe, axis=0)
        return qvecs_qe, ranks_aqe

    if RW == True: 
    # For database with less than 120,000 images, we use this setting to improve accuracy.
       T_qe_1=time.time()
       k = 10 # k refers to top k results in preliminary ranks
       w = 8. / 2
       it_times = 3
       
       qvecs_qe, ranks_aqe = feature_enhancement(it_times, k, ranks, vecs, w)
       T_qe_2=time.time()
    #    print('mAP after Enhancement')
    #    compute_map_and_print2(dataset, ranks_aqe, gnd) 
    #    print('Time for Enhancement:', ((T_qe_2-T_qe_1)/query_num)) 
    #    print('----------------------------------------')
    
       '''
       For small and medium databases, we can increase kg to obtain higher accuracy
       Generally, we can set k_gallery = 50, 100 or 200
       '''
       truncation_number = 2000
       k_gallery = 200 
       k_query = 3
       T_dfs_1 = time.time()
       diffusion = Diffusion(vecs.T, cache_dir)
       offline = diffusion.get_offline_results(truncation_number, k_gallery)
       T_dfs_search_1 = time.time()
       print('Re-search')
       if AQE == True:
             sims, idx = diffusion.knn.search(qvecs_qe.T, k_query)
       else:
             sims, idx = diffusion.knn.search(qvecs.T, k_query)
       sims = sims ** 3
       q_num_dfs = idx.shape[0]

       print('Linear combination')
       truncation_scores = np.empty((q_num_dfs, truncation_number), dtype=np.float32)
       truncation_ranks = np.empty((q_num_dfs, truncation_number), dtype=np.int)
       for i in tqdm(range(q_num_dfs), desc='searching query'):
            scores = sims[i] @ offline[idx[i]]
            parts = np.argpartition(-scores, truncation_number)[:truncation_number]
            ranks = np.argsort(-scores[parts])
            truncation_scores[i] = scores[parts][ranks]
            truncation_ranks[i] = parts[ranks]
       T_dfs_search_2 = time.time()
       print('Search: search costs {}s'.format((T_dfs_search_2 - T_dfs_search_1)/query_num))
       ranks_dfs = truncation_ranks.T
       T_dfs_2 = time.time()
       gnd_name2 = os.path.splitext(os.path.basename(gnd_path2))[0]
       with open(gnd_path2, 'rb') as f:
             gnd2 = pickle.load(f)['gnd']
       print('mAP after Enhancement (Random Walk)')
       compute_map_and_print2(gnd_name2.split("_")[-1], ranks_dfs, gnd2)
       print('Time for Enhancement (Random Walk):', ((T_dfs_2-T_dfs_1)/query_num))
       print('Time for QGE:', ((T_dfs_2-T_qe_1)/query_num))
       print('-----------------------------------------------------')
       '''
       The following two lines are used for removing cache. Saving the cache can improve speed.
       If we need to change parameters, we need to remove the cache and rebuild new cache.
       '''
       # offline_file_path = '/users/qzhang7/src_qz/4_src-base/diffusion/tmp/' + dataset + '/offline.jbl'
       # os.remove(offline_file_path)
    
    else: 
    # For database with more than 120,000 images, we use this setting to achieve the accuracy-speed balance
       T_qe_1= time.time()
       k = 3
       w = 8. / 2
       it_times = 1
       
       qvecs_qe, ranks_aqe = feature_enhancement(it_times, k, ranks, qvecs, vecs, w)
       T_qe_2=time.time()
       print('mAP after Enhancement')
       compute_map_and_print2(dataset, ranks_aqe, gnd) 
       print('Time for Enhancement:', ((T_qe_2-T_qe_1)/query_num)) 
       print('----------------------------------------')
        
def qge1(ranks, qvec, vecs, K):
    def feature_enhancement(it_times, k, ranks, qvecs, vecs, w):
        for it_time in range(it_times):
            qe_weight = (np.arange(k, 0, -1) / k).reshape(1, k, 1) # build an array, [1, 1/2, ..., 1/k]
            ranks_top = ranks[:k, int(0): int(ranks.shape[1])]
            top_k_vecs = vecs[:, ranks_top]
            # If we have query images in databases, we can use the following line.
            qvecs_top = (top_k_vecs * (qe_weight ** w)).sum(axis=1)
            # If we don't have query images in databases, we can use the following line.
            # qvecs_top = (top_k_vecs * (qe_weight ** w)).sum(axis=1) + qvecs
            qvecs_top = qvecs_top / (np.linalg.norm(qvecs_top, ord=2, axis=0, keepdims=True) + 1e-6)
            qvecs_qe = qvecs_top
            scores_aqe = np.dot(vecs.T, qvecs_qe)
            ranks_aqe = np.argsort(-scores_aqe, axis=0)
        return qvecs_qe, ranks_aqe
    k = 3 
    w = 8. / 2
    it_times = 1 
    qvecs_qe, ranks_aqe = feature_enhancement(it_times, k, ranks, vecs, w)
    return ranks_aqe

'''
5 Average Query Expansion
This is the original query expansion. This is accurate. But it is slower than QGE.

url: https://github.com/fuxinjiang/huawei2020/blob/6eaffa9f18732a27a29204834b22b57d9817f08e/utils/db_qe.py
'''
def average_query_expansion(qvecs,vecs,K,dataset,gnd):
    def _centerize(v1, v2):
        concat = np.concatenate([v1, v2], axis=0)
        center = np.mean(concat, axis=0)
        return v1-center, v2-center

    def _l2_normalize(v):
        norm = np.expand_dims(np.linalg.norm(v, axis=1), axis=1)
        if np.any(norm == 0):
            return v
        return v / norm

    def postprocess(query_vecs, reference_vecs):
        # centerize
        query_vecs, reference_vecs = _centerize(query_vecs, reference_vecs)
        # l2 normalization
        query_vecs = _l2_normalize(query_vecs)
        reference_vecs = _l2_normalize(reference_vecs)
        return query_vecs, reference_vecs

    def calculate_sim_matrix(query_vecs, reference_vecs):
        query_vecs, reference_vecs = postprocess(query_vecs, reference_vecs)
        # speed up
        return 2-2*np.dot(query_vecs, reference_vecs.T)

    print('Average Query Expansion')
    
    Tqe2_1 = time.time()
    top_k = 3
    sim_mat = calculate_sim_matrix(qvecs.T, vecs.T)
    indices = np.argsort(sim_mat, axis=1)

    top_k_ref_mean = np.mean(vecs.T[indices[:, :top_k], :], axis=1)
    qvecs = np.concatenate([qvecs.T, top_k_ref_mean], axis=1)
    print('Reference Augmentation')
    # Reference augmentation
    sim_mat = calculate_sim_matrix(vecs.T, vecs.T)
    indices = np.argsort(sim_mat, axis=1)

    top_k_ref_mean = np.mean(vecs.T[indices[:, 1:top_k+1], :], axis=1)
    vecs = np.concatenate([vecs.T, top_k_ref_mean], axis=1)

    match_idx, time_per_query = matching_L2(K, vecs, qvecs)
    Tqe2_2 = time.time()
    print(vecs.shape)
    print(qvecs.shape)
    ranks_qe2 = match_idx.T
    print('------------------------------------------------------')
    print('mAP after qe2:')
    compute_map_and_print2(dataset, ranks_qe2, gnd)
    print('Time for Query Expansion 2:', ((Tqe2_2-Tqe2_1)/qvecs.shape[0]))
    print('------------------------------------------------------')

'''
6 Database Augmentation

This another global feature-based re-ranking method, database augmentation. This method is accurate.
But it is slower than QGE. Maybe there is a way to improve it.

url: https://github.com/fuxinjiang/huawei2020/blob/6eaffa9f18732a27a29204834b22b57d9817f08e/utils/db_qe.py
'''
def database_augmentation(qvecs,vecs,K,dataset,gnd):
    def _centerize(v1, v2):
        concat = np.concatenate([v1, v2], axis=0)
        center = np.mean(concat, axis=0)
        return v1-center, v2-center

    def _l2_normalize(v):
        norm = np.expand_dims(np.linalg.norm(v, axis=1), axis=1)
        if np.any(norm == 0):
            return v
        return v / norm

    def postprocess(query_vecs, reference_vecs):
        """
        Postprocessing:
        1) Moving the origin of the feature space to the center of the feature vectors
        2) L2-normalization
        """
        # centerize
        query_vecs, reference_vecs = _centerize(query_vecs, reference_vecs)
        # l2 normalization
        query_vecs = _l2_normalize(query_vecs)
        reference_vecs = _l2_normalize(reference_vecs)
        return query_vecs, reference_vecs

    def calculate_sim_matrix(query_vecs, reference_vecs):
        query_vecs, reference_vecs = postprocess(query_vecs, reference_vecs)
        return 2-2*np.dot(query_vecs, reference_vecs.T)

    # Database augmentation
    top_k = 3

    print('Database Augmentation')
    Tdba1 = time.time()
    weights = np.logspace(0, -2., top_k+1)
    # Query augmentation
    print('Query Augmentation')
    sim_mat = calculate_sim_matrix(qvecs.T, vecs.T)
    indices = np.argsort(sim_mat, axis=1)
    top_k_ref = vecs.T[indices[:, :top_k], :]
    qvecs = np.tensordot(weights, np.concatenate([np.expand_dims(qvecs.T, 1), top_k_ref], axis=1), axes=(0, 1))
    print('Reference Augmentation')
    # Reference augmentation
    sim_mat = calculate_sim_matrix(vecs.T, vecs.T)
    indices = np.argsort(sim_mat, axis=1)

    top_k_ref = vecs.T[indices[:, :top_k+1], :]
    vecs = np.tensordot(weights, top_k_ref, axes=(0, 1))
    print(qvecs.shape)
    print(vecs.shape)
    match_idx, time_per_query = matching_L2(K, vecs, qvecs)
    Tdba2 = time.time()
    ranks_dba = match_idx.T
    print('------------------------------------------------------')
    print('mAP after dba:')
    compute_map_and_print2(dataset, ranks_dba, gnd)
    print('Time for Database Augmentation:', ((Tdba2-Tdba1)/qvecs.shape[0]))
    print('------------------------------------------------------')

'''
benchmarks
'''

'''
7 kr re-ranking
This method is a global feature-based reranking method. It is popular. Some pipelines implement it 
as re-ranking. Compared to QGE, it is a little slower. Its accuracy is good.

CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
'''
def kr_reranking(qvecs, vecs, 
                #  dataset, gnd  #If you want to test the custom dataset, you need to commnet out 
                # these two variables
                 ):
    def euclidean_distance(qf, gf):
        m = qf.shape[0]
        n = gf.shape[0]

        # dist_mat = torch.pow(qf,2).sum(dim=1, keepdim=True).expand(m,n) +\
        #     torch.pow(gf,2).sum(dim=1, keepdim=True).expand(n,m).t()
        # dist_mat.addmm_(1,-2,qf,gf.t())

        # For L2-norm feature
        dist_mat = 2 - 2 * torch.matmul(qf, gf.t())
        return dist_mat

    def batch_euclidean_distance(qf, gf, N=6000):
        m = qf.shape[0]
        n = gf.shape[0]

        dist_mat = []
        for j in range(n // N + 1):
            temp_gf = gf[j * N:j * N + N]
            temp_qd = []
            for i in range(m // N + 1):
                temp_qf = qf[i * N:i * N + N]
                temp_d = euclidean_distance(temp_qf, temp_gf)
                temp_qd.append(temp_d)
            temp_qd = torch.cat(temp_qd, dim=0)
            temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
            dist_mat.append(temp_qd.t().cpu())
        del temp_qd
        del temp_gf
        del temp_qf
        del temp_d
        torch.cuda.empty_cache()  # empty GPU memory
        dist_mat = torch.cat(dist_mat, dim=0)
        return dist_mat

    # Compute top results
    def batch_torch_topk(qf, gf, k1, N=6000):
        m = qf.shape[0]
        n = gf.shape[0]

        dist_mat = []
        initial_rank = []
        for j in range(n // N + 1):
            temp_gf = gf[j * N:j * N + N]
            temp_qd = []
            for i in range(m // N + 1):
                temp_qf = qf[i * N:i * N + N]
                temp_d = euclidean_distance(temp_qf, temp_gf)
                temp_qd.append(temp_d)
            temp_qd = torch.cat(temp_qd, dim=0)
            temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
            temp_qd = temp_qd.t()
            initial_rank.append(torch.topk(temp_qd, k=k1, dim=1, largest=False, sorted=True)[1])

        del temp_qd
        del temp_gf
        del temp_qf
        del temp_d
        torch.cuda.empty_cache()  # empty GPU memory
        initial_rank = torch.cat(initial_rank, dim=0).cpu().numpy()
        return initial_rank

    def batch_v(feat, R, all_num):
        V = np.zeros((all_num, all_num), dtype=np.float32)
        m = feat.shape[0]
        for i in tqdm(range(m)):
            temp_gf = feat[i].unsqueeze(0)
            # temp_qd = []
            temp_qd = euclidean_distance(temp_gf, feat)
            temp_qd = temp_qd / (torch.max(temp_qd))
            temp_qd = temp_qd.squeeze()
            temp_qd = temp_qd[R[i]]
            weight = torch.exp(-temp_qd)
            weight = (weight / torch.sum(weight)).cpu().numpy()
            V[i, R[i]] = weight.astype(np.float32)
        return V

    def k_reciprocal_neigh(initial_rank, i, k1):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        return forward_k_neigh_index[fi]

    '''
    kr main
    '''
    Tkr1 = time.time()
    k1 = 20
    k2 = 6
    lambda_value = 0.3

    probFea = torch.tensor(qvecs.T, dtype=torch.float32)
    galFea = torch.tensor(vecs.T, dtype=torch.float32)

    t1 = time.time()
    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]
    feat = torch.cat([probFea, galFea]).cuda()
    initial_rank = batch_torch_topk(feat, feat, k1 + 1, N=6000)
    # del feat
    del probFea
    del galFea
    torch.cuda.empty_cache()  # empty GPU memory
    gc.collect()  # empty memory
    print('Using totally {:.2f}s to compute initial_rank'.format(time.time() - t1))
    print('starting re_ranking')

    R = []
    for i in tqdm(range(all_num)):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                   candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        R.append(k_reciprocal_expansion_index)

    gc.collect()  # empty memory
    print('Using totally {:.2f}S to compute R'.format(time.time() - t1))
    V = batch_v(feat, R, all_num)
    del R
    gc.collect()  # empty memory
    print('Using totally {:.2f}S to compute V-1'.format(time.time() - t1))
    initial_rank = initial_rank[:, :k2]

    if k2 != 1:
         V_qe = np.zeros_like(V, dtype=np.float16)
         for i in range(all_num):
             V_qe[i, :] = np.mean(V[initial_rank[i], :], axis=0)
         V = V_qe
         del V_qe
    del initial_rank

    gc.collect()  # empty memory
    print('Using totally {:.2f}S to compute V-2'.format(time.time() - t1))
    invIndex = []

    for i in range(all_num):
          invIndex.append(np.where(V[:, i] != 0)[0])
    print('Using totally {:.2f}S to compute invIndex'.format(time.time() - t1))

    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)
    for i in tqdm(range(query_num)):
          temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
          indNonZero = np.where(V[i, :] != 0)[0]
          indImages = [invIndex[ind] for ind in indNonZero]
          for j in range(len(indNonZero)):
              temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                           V[indImages[j], indNonZero[j]])
          jaccard_dist[i] = 1 - temp_min / (2. - temp_min)
    del V
    gc.collect()  # empty memory
    original_dist = batch_euclidean_distance(feat, feat[:query_num, :]).numpy()
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist

    del jaccard_dist

    final_dist = final_dist[:query_num, query_num:]
    print('Using totally {:.2f}S to compute final_distance'.format(time.time() - t1))

    indices = np.argsort(final_dist, axis=1)
    ranks_reranking = indices.T
    Tkr2 = time.time()
    print('------------------------------------------------------')
    print('mAP after reranking:')
    # compute_map_and_print2(dataset, ranks_reranking, gnd)
    print('Time for kr:', ((Tkr2-Tkr1)/qvecs.shape[1]))
    print('------------------------------------------------------')
    return indices

'''
8 ransac: sift-adalam
This method is a traditional local feature-based re-ranking method. We can use SIFT to extract features
and use AdaLAM to filter and match. There is a ransac part in AdaLAM. This method is a little better than
SIFT-BF. BF is brute force matching. AdaLAM is faster and more accurate. But there is some drawbacks in
this method: it is not fast enough. And its performance is not very good under complicated illumination
situations. Therefore, we develop SAHA. We use AffNet and HardNet to improve features.

This method is a benchmark method.
'''
def ransac_sift(query_num, qimages, images, ranks, dataset, gnd):
    
    device = torch.device('cpu')
    try:
         if torch.cuda.is_available():
             device = torch.device('cuda:0')
             print ("GPU mode")
    except:
         print ('CPU mode')

    def bubbleSort(nums):
        for i in range(nums.shape[0]-1):
             for j in range(nums.shape[0]-i-1):
                 if nums[j,1] <= nums[j+1,1]:
                     nums[j,1], nums[j+1,1] = nums[j+1,1], nums[j,1]
                     nums[j,0], nums[j+1,0] = nums[j+1,0], nums[j,0]
        return nums
    
    b = 30 # this is the top-k preliminary results in gallery for re-ranking
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.FlannBasedMatcher_create()

    T_ransac_sift_1 = time.time()
    for i in tqdm(range(0,query_num)): 
         a = np.zeros(shape=(b, 2))
         ranks_query_index = int(i)
         query_ab_path = qimages[ranks_query_index]
         query_image = cv2.imread(query_ab_path)
         query_image = cv2.resize(query_image, (1000, 1000))
         keypoints_q, descriptors_q = sift.detectAndCompute(query_image, mask=np.ones(shape=query_image.shape[:-1] + (1,), dtype=np.uint8))

         for j in range(0,b):
            ranks_image_index = int(ranks[j,i])
            image_ab_path = images[ranks_image_index]
            gallery_image = cv2.imread(image_ab_path)
            gallery_image = cv2.resize(gallery_image, (1000, 1000))
            keypoints_g, descriptors_g = sift.detectAndCompute(gallery_image, mask=np.ones(shape=gallery_image.shape[:-1] + (1,), dtype=np.uint8))

            # 1 BF match
            # matches = bf.match(descriptors_q, descriptors_g)
            # matches = sorted(matches, key=lambda x: x.distance)
            # # kp_ratio = []
            # # for j in range(len(matches)):
            # #     if matches[j].distance < 200:
            # #         kp_ratio.append(matches[j])

            # # if len(kp_ratio) > 10:
            # #     ptsA = np.float32([keypoints_q[m.queryIdx].pt for m in kp_ratio]).reshape(-1, 1, 2)
            # #     ptsB = np.float32([keypoints_g[m.trainIdx].pt for m in kp_ratio]).reshape(-1, 1, 2)

            # #     _, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold=4)

            # #     spatial_score.append(status.sum())
            # # else:
            # #     spatial_score.append(2)

            # # spatial_score = np.argsort(-np.array(spatial_score))

            # a[j, 0] = ranks_image_index
            # a[j, 1] = len(matches)

            # 2 RANSAC adalam match
            points1 = np.array([k.pt for k in keypoints_q], dtype=np.float32)
            ors1 = np.array([k.angle for k in keypoints_q], dtype=np.float32)
            scs1 = np.array([k.size for k in keypoints_q], dtype=np.float32)
            points2 = np.array([k.pt for k in keypoints_g], dtype=np.float32)
            ors2 = np.array([k.angle for k in keypoints_g], dtype=np.float32)
            scs2 = np.array([k.size for k in keypoints_g], dtype=np.float32)

            matcher = AdalamFilter()
            matches = matcher.match_and_filter(k1=points1, k2=points2,
                                               o1=ors1, o2=ors2,
                                               d1=descriptors_q, d2=descriptors_g,
                                               s1=scs1, s2=scs2,
                                               im1shape=query_image.shape[:2], im2shape=gallery_image.shape[:2]).cpu().numpy()

            a[j, 0] = ranks_image_index
            a[j, 1] = len(matches)

        # Bubble Sort
         a = bubbleSort(a)
         for p in range(0, b):
            ranks[p,i]=a[p,0]    
    compute_map_and_print(dataset, ranks, gnd) 
    T_ransac_sift_2 = time.time()
    print('Time for SIFT reranking per query:%s' % ((T_ransac_sift_2-T_ransac_sift_1)/query_num))
