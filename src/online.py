import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import pickle
import warnings
import numpy as np
from PIL import Image
from torchvision import transforms
from src.networks.imageretrievalnet import init_network, extract_vectors, extract_vectors_single
from datetime import datetime as dt
from flask import Flask, request, render_template
from pathlib import Path
from src.datasets.testdataset import configdataset
from src.utils.nnsearch import *
from src.utils.Reranking import *
from src.utils.networks import load_network
from src.utils.general import load_path_features


datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k', 'revisitop1m']

# test options
parser = argparse.ArgumentParser(description='Historical Image Retrieval')
parser.add_argument('--network', '-n', metavar='NETWORK', default='resnet101-solar-best.pth', 
                    help="network to be evaluated. ")
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='roxford5k,rparis6k',
                    help="comma separated list of test datasets: " +
                        " | ".join(datasets_names) +
                        " (default: 'roxford5k,rparis6k')")
parser.add_argument('--K-nearest-neighbour', '-K', default=30, type=int, metavar='K',
                    help="retreive top-K results (default: 30)")
parser.add_argument('--matching_method', '-mm', default='L2', help="select matching methods: L2, PQ, ANNOY, HNSW, PQ_HNSW")
parser.add_argument('--ifgenerate', '-gen', dest='ifgenerate', action='store_true',
                    help='Include --ifgenerate if the trees/graphs/distance tables have not been generated and saved')
parser.add_argument('--image-size', '-imsize', dest='image_size', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1, 2**(1/2), 1/2**(1/2)]',
                    help="use multiscale vectors for testing, " +
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--soa', action='store_true',
                    help='use soa blocks')
parser.add_argument('--soa-layers', type=str, default='45',
                    help='config soa blocks for second-order attention')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")

warnings.filterwarnings("ignore", category=UserWarning)

# parse the arguments
args = parser.parse_args()

app = Flask(__name__)

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
    transforms.Resize(args.image_size),
    transforms.ToTensor(),
    normalize
])


#############
# RETRIEVAL
#############
# Read image features
datasets = args.datasets.split(',')
dim_vec = 2048
vecs = np.empty((dim_vec, 0))
img_paths = []
for dataset in datasets:
    vecs_temp, img_r_path_temp = load_path_features(dataset)
    vecs = np.concatenate([vecs, vecs_temp], axis=1)
    images = ['/static/' + i for i in img_r_path_temp]
    img_paths = img_paths + images

rel_img_paths = [os.path.relpath(path, '/static/test/') for path in img_paths]
K = args.K_nearest_neighbour

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        if not os.path.exists("src/static/uploaded/"):
            os.makedirs("src/static/uploaded/")
        uploaded_img_path = "src/static/upload/" + dt.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        query_path = '/' + '/'.join(uploaded_img_path.split('/')[1:])
        print(query_path)
        qvec = extract_vectors_single(net, uploaded_img_path, args.image_size, transform, ms=ms)
        qvec = np.expand_dims(qvec.numpy(), axis=1)

        # Run search
        # Select methods for preliminary ranking
        # Set ifgenerate=True for the first time to build trees/graphs or to do quantization
        # Then the generated trees, graphs, etc will be saved
        # Set ifgenerate=False to skip all the preparations in later use
        # Note: parameters like N_books, n_bits_perbook, n_trees, etc will significantly influence the retrieval performance and efficiency
        # Usually larger values lead to better performance but slower retrieval time
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

        
        # Re-ranking
        ranks = match_idx.T
        ranks2 = qge1(ranks, qvec, vecs, K)
        idx2 = ranks2.T

        # scores1 = [(id, img_paths[id]) for id in np.squeeze(match_idx)[:10]]
        scores2 = [(rel_img_paths[id], img_paths[id]) for id in np.squeeze(idx2)[:K]]
        return render_template('index.html', 
                               query_path=query_path,
                            #    scores=scores1,
                               marks=scores2)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
    # app.run(host="localhost", port=8000, debug=True)
