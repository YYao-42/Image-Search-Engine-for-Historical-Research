import os
import hashlib
import pickle
import fnmatch
from torch.utils.tensorboard import SummaryWriter


def get_root():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


def get_data_root():
    return os.path.join(get_root(), 'data')


def htime(c):
    c = round(c)
    
    days = c // 86400
    hours = c // 3600 % 24
    minutes = c // 60 % 60
    seconds = c % 60

    if days > 0:
        return '{:d}d {:d}h {:d}m {:d}s'.format(days, hours, minutes, seconds)
    if hours > 0:
        return '{:d}h {:d}m {:d}s'.format(hours, minutes, seconds)
    if minutes > 0:
        return '{:d}m {:d}s'.format(minutes, seconds)
    return '{:d}s'.format(seconds)


def sha256_hash(filename, block_size=65536, length=8):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()[:length-1]

def tb_setup(save_dir):
    # Setup for tensorboard
    tb_save_dir = os.path.join(
                    save_dir,
                    'summary',
                    )
    if not os.path.exists(tb_save_dir):
        os.makedirs(tb_save_dir)
    
    trash_list = os.listdir(tb_save_dir)
    for entry in trash_list:
        filename = os.path.join(tb_save_dir, entry)
        if fnmatch.fnmatch(entry, '*tfevents*'):
            os.remove(filename)

    summary = SummaryWriter(log_dir=tb_save_dir)

    return summary

def path_all_jpg(directory, start):
    paths = []
    for dirpath, _, filenames in os.walk(directory):
        paths = paths + [os.path.join(dirpath, f) for f in filenames if f.endswith(".jpg")]
        # paths = paths + [os.path.join(dirpath, f) for f in filenames]
    rel_paths = [os.path.relpath(path, start) for path in paths]
    return paths, rel_paths

def save_path_feature(dataset, vecs, img_r_path):
    # save the dictionary of paths and features of images into a pkl file    
    isExist = os.path.exists('outputs')
    if not isExist:
        os.makedirs('outputs')
    
    path_feature = {}
    path_feature['path'] = img_r_path
    path_feature['feature'] = vecs

    if '/' in dataset:
        dataset = dataset.replace('/', '_')
    file_path_feature = 'outputs/features/' + dataset  +'_path_feature_multi2.pkl'
    afile = open(file_path_feature, "wb")
    pickle.dump(path_feature, afile)
    afile.close()

def load_path_features(dataset):
    if '/' in dataset:
        dataset = dataset.replace('/', '_')
    file_path_feature = 'outputs/features/' + dataset + '_path_feature_single.pkl'
    with open(file_path_feature, 'rb') as pickle_file:
        path_feature = pickle.load(pickle_file)
    vecs = path_feature['feature']
    img_r_path = path_feature['path']
    return vecs, img_r_path