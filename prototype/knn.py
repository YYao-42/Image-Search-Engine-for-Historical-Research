import os
import numpy as np
import time
import faiss
from tqdm import tqdm


class BaseKNN(object):
    def __init__(self, database, method):
        if database.dtype != np.float32:
            database = database.astype(np.float32)
        self.N = len(database)
        self.D = database[0].shape[-1]
        self.database = database if database.flags['C_CONTIGUOUS'] \
                               else np.ascontiguousarray(database)

    def add(self, batch_size=10000):
        if self.N <= batch_size:
            self.index.add(self.database)
        else:
            [self.index.add(self.database[i:i+batch_size])
                    for i in tqdm(range(0, len(self.database), batch_size),
                                  desc='index adding')]

    def search(self, queries, k):
        if not queries.flags['C_CONTIGUOUS']:
            queries = np.ascontiguousarray(queries)
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        sims, ids = self.index.search(queries, k)
        return sims, ids

class KNN(BaseKNN):
    def __init__(self, database, method):
        super().__init__(database, method)
        self.index = {'cosine': faiss.IndexFlatIP,
                      'euclidean': faiss.IndexFlatL2}[method](self.D)
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.add()


class ANN(BaseKNN):
    def __init__(self, database, method, M=128, nbits=8, nlist=316, nprobe=64):
        super().__init__(database, method)
        self.quantizer = {'cosine': faiss.IndexFlatIP,
                          'euclidean': faiss.IndexFlatL2}[method](self.D)
        self.index = faiss.IndexIVFPQ(self.quantizer, self.D, nlist, M, nbits)
        samples = database[np.random.permutation(np.arange(self.N))[:self.N // 5]]
        print("ANN processing")
        self.index.train(samples) #!!!
        self.add()
        self.index.nprobe = nprobe #!!!

