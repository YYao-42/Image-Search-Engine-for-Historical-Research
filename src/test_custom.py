import os
import os.path
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.utils.nnsearch import *
from src.utils.evaluate import mAP_custom

def path_all_jpg(directory, start):
    paths = []
    for dirpath, _, filenames in os.walk(directory):
        paths = paths + [os.path.join(dirpath, f) for f in filenames if f.endswith(".jpg")]
        # paths = paths + [os.path.join(dirpath, f) for f in filenames]
    rel_paths = [os.path.relpath(path, start) for path in paths]
    return paths, rel_paths

def load_path_features(dataset):
    if '/' in dataset:
        dataset = dataset.replace('/', '_')
    file_path_feature = 'outputs/features/' + dataset + '_path_feature_2.pkl'
    with open(file_path_feature, 'rb') as pickle_file:
        path_feature = pickle.load(pickle_file)
    vecs = path_feature['feature']
    img_r_path = path_feature['path']
    return vecs, img_r_path

Custom_q, relpaths_q = load_path_features('custom/query')
Custom_d, relpaths_d = load_path_features('custom/database')

n_database = Custom_d.shape[1]
K = n_database
match_idx, time_per_query = matching_L2(K, Custom_d.T, Custom_q.T)
mAP = mAP_custom(K, match_idx, relpaths_q, relpaths_d)
print('mean average precision: ', mAP)

# Output ranked images
rank_res = {}
for i in range(len(relpaths_q)):
    rank_res[relpaths_q[i]] = [relpaths_d[j] for j in match_idx[i,:]]
file_rankres = 'outputs/ranks/'  + 'custom_ranking_result.pkl'
a_file = open(file_rankres, "wb")
pickle.dump(rank_res, a_file)
a_file.close()

# Visualization
folder_path_q = os.path.join('/home/yananhu/SOLAR/data/test', 'custom/query')
qimages, img_r_path = path_all_jpg(folder_path_q, start="/home/yananhu/SOLAR")
folder_path_d = os.path.join('/home/yananhu/SOLAR/data/test', 'custom/database')
images, img_r_path = path_all_jpg(folder_path_d, start="/home/yananhu/SOLAR")
K_show = 10
idx_select = 3
query_image = qimages[idx_select]
matching_images = [images[j] for j in match_idx[idx_select, :]]
plt.close('all')
plt.figure(figsize=(10, 4), dpi=80)
ax = plt.subplot2grid((2, K_show), (0, 0))
ax.axis('off')
ax.set_title('Query')
img = mpimg.imread(query_image)
plt.imshow(img)
for j in range(K_show):
        #     if dataset == 'oxford5k' or dataset == 'paris6k':
        #         if np.in1d(match_idx[idx_select, i], gnd[idx_select]['ok'])[0]:
        #             plt.rcParams["axes.edgecolor"] = "green"
        #         else:
        #             plt.rcParams["axes.edgecolor"] = "red"
        #     if dataset == 'roxford5k' or dataset == 'rparis6k':
        #         if np.in1d(match_idx[idx_select, i], gnd[idx_select]['easy'])[0]:
        #             plt.rcParams["axes.edgecolor"] = "green"
        #         elif np.in1d(match_idx[idx_select, i], gnd[idx_select]['hard'])[0]:
        #             plt.rcParams["axes.edgecolor"] = "blue"
        #         elif np.in1d(match_idx[idx_select, i], gnd[idx_select]['junk'])[0]:
        #             plt.rcParams["axes.edgecolor"] = "red"
    plt.rcParams["axes.linewidth"] = 2.50
    ax = plt.subplot2grid((2, K_show), (1, j))
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_title('#' + str(j + 1))
    img = mpimg.imread(matching_images[j])
    plt.imshow(img)
plt.tight_layout(pad=0.5)
file_vis_path = 'outputs/ranks/custom/'  + str(idx_select) + '_vis.png'
plt.savefig(file_vis_path)

print('end')
