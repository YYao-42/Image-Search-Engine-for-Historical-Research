'''
This is the offline part of SAHA. We use this to store offline files.
By this file, we can get features of all images in databases. Therefore, this process costs a lot of time.
But this saves our time on the online step.
'''

'''
1 import
'''
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from tqdm import tqdm
import xlrd
import cv2
import time

import torch
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *

from adalam import AdalamFilter

'''
2 build SAHA settings
'''
gnds = xlrd.open_workbook("gnd_rparis6k.xlsx")
gnd = gnds.sheet_by_index(0)

device = torch.device('cpu')
try:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print ("GPU mode")
except:
    print ('CPU mode')

def convert_kpts(cv2_kpts):
    keypoints = np.array([(x.pt[0], x.pt[1]) for x in cv2_kpts ]).reshape(-1, 2)
    scales = np.array([12.0* x.size for x in cv2_kpts ]).reshape(-1, 1)
    angles = np.array([x.angle for x in cv2_kpts ]).reshape(-1, 1)
    responses = np.array([x.response for x in cv2_kpts]).reshape(-1, 1)
    return keypoints, scales, angles, responses

sift_det = cv2.xfeatures2d.SIFT_create(8000, contrastThreshold=-10000, edgeThreshold=-10000)
hardnet8 = KF.HardNet8(True).eval().to(device)
affnet = KF.LAFAffNetShapeEstimator(True).eval().to(device)
matcher = AdalamFilter()

'''
3 read images and extract features
'''
'''
By the following loop, we can store query features in a specific address.
'''
for i in tqdm(range(0,70)): # 70 is the query number
    gnd_query_name = gnd.cell(3, i+1).value
    gnd_query_path = '/home/qzhang7/Datasets/rparis6k/jpg/' + str(gnd_query_name) + '.jpg'
    query_image = cv2.imread(gnd_query_path, cv2.COLOR_BGR2RGB)
    query_image = cv2.resize(query_image, (1000, 1000)) # this is the resolution of images
    h1, w1 = query_image.shape[:2]
    query_h_path = '/home/qzhang7/SIFT_features/rparis/query1000/' +\
                    str(gnd_query_name) + '_h1' + '.npy'
    query_w_path = '/home/qzhang7/SIFT_features/rparis/query1000/' +\
                    str(gnd_query_name) + '_w1' + '.npy'
    np.save(query_h_path, h1)
    np.save(query_w_path, w1)

    keypoints_query = sift_det.detect(query_image, None)[:8000] 

    with torch.no_grad():
        timg1 = K.image_to_tensor(query_image, False).float() / 255.
        timg1 = timg1.to(device)
        timg_gray1 = K.rgb_to_grayscale(timg1)

        lafs1 = laf_from_opencv_SIFT_kpts(keypoints_query, device=device)
        lafs_new1 = affnet(lafs1, timg_gray1)

        patches = KF.extract_patches_from_pyramid(timg_gray1, lafs_new1, 32)
        B1, N1, CH1, H1, W1 = patches.size()
        descriptors_query = hardnet8(patches.view(B1 * N1, CH1, H1, W1)).view(B1 * N1, -1).detach().cpu().numpy()
        query_descriptors_path = '/home/qzhang7/SIFT_features/rparis/query1000/'\
                                 + str(gnd_query_name) + '_descriptors' + '.npy'
        np.save(query_descriptors_path, descriptors_query)

    kp_q, s_q, a_q, r_q = convert_kpts(keypoints_query)
    query_kp_path = '/home/qzhang7/SIFT_features/rparis/query1000/' +\
                    str(gnd_query_name) + '_kp' + '.npy'
    query_s_path = '/home/qzhang7/SIFT_features/rparis/query1000/' +\
                    str(gnd_query_name) + '_s' + '.npy'
    query_a_path = '/home/qzhang7/SIFT_features/rparis/query1000/' +\
                    str(gnd_query_name) + '_a' + '.npy'
    # query_r_path = '/home/qzhang7/SIFT_features/rparis/query/' +\
    #                 str(gnd_query_name) + '_r' + '.npy'
    np.save(query_kp_path, kp_q)
    np.save(query_s_path, s_q)
    np.save(query_a_path, a_q)
    # np.save(query_r_path, r_q)

'''
By the following loop, we can store gallery features in a specific address. 
'''    
for j in tqdm(range(0, 6322)): # 6322 is the gallery number
    gnd_image_name = gnd.cell(2, j+1).value
    gnd_image_path = '/home/qzhang7/Datasets/rparis6k/jpg/' + str(gnd_image_name) + '.jpg'
    image = cv2.imread(gnd_image_path, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (800, 600)) # this is the resolution of images
    h2, w2 = image.shape[:2]

    image_h_path = '/home/qzhang7/SIFT_features/rparis/gallery1000/' +\
                   str(gnd_image_name) + '_h2' + '.npy'
    image_w_path = '/home/qzhang7/SIFT_features/rparis/gallery1000/' +\
                   str(gnd_image_name) + '_w2' + '.npy'
    np.save(image_h_path, h2)
    np.save(image_w_path, w2)

    keypoints_image = sift_det.detect(image, None)[:8000]

    with torch.no_grad():
        timg2 = K.image_to_tensor(image, False).float() / 255.
        timg2 = timg2.to(device)
        timg_gray2 = K.rgb_to_grayscale(timg2)

        lafs2 = laf_from_opencv_SIFT_kpts(keypoints_image, device=device)
        lafs_new2 = affnet(lafs2, timg_gray2)

        patches = KF.extract_patches_from_pyramid(timg_gray2, lafs_new2, 32)
        B2, N2, CH2, H2, W2 = patches.size()
        descriptors_image = hardnet8(patches.view(B2 * N2, CH2, H2, W2)).view(B2 * N2, -1).detach().cpu().numpy()
        image_descriptors_path = '/home/qzhang7/SIFT_features/rparis/gallery1000/'\
                                 + str(gnd_image_name) + '_descriptors' + '.npy'
        np.save(image_descriptors_path, descriptors_image)
    kp_i, s_i, a_i, r_i = convert_kpts(keypoints_image)

    image_kp_path = '/home/qzhang7/SIFT_features/rparis/gallery1000/' +\
                    str(gnd_image_name) + '_kp' + '.npy'
    image_s_path = '/home/qzhang7/SIFT_features/rparis/gallery1000/' +\
                   str(gnd_image_name) + '_s' + '.npy'
    image_a_path = '/home/qzhang7/SIFT_features/rparis/gallery1000/' +\
                   str(gnd_image_name) + '_a' + '.npy'
    # image_r_path = '/home/qzhang7/SIFT_features/rparis/gallery/' +\
    #                str(gnd_image_name) + '_r' + '.npy'
    np.save(image_kp_path, kp_i)
    np.save(image_s_path, s_i)
    np.save(image_a_path, a_i)
    # np.save(image_r_path, r_i)