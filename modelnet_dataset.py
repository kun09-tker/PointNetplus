import numpy as np
import os
import pandas as pd
from lib import provider

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNetDataset():
    def __init__(self, root, batch_size = 32, npoints = 1024, normalize=True, normal_channel=False, cache_size=15000, shuffle=True):
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        self.shuffle = shuffle
        # if modelnet10:
        #     self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        # else:
        #     self.catfile = os.path.join(self.root, 'shape_names.txt')
        self.cat = ['nofight', 'fight']
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        self.normal_channel = normal_channel
        
        # shape_ids = {}
        # if modelnet10:
        #     shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))] 
        #     shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        # else:
        #     shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))] 
        #     shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        # assert(split=='train' or split=='test')
        # shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = []
        for c in self.cat:
            for dirname, _ , filenames in os.walk(os.path.join(self.root,c)):
                for filename in filenames:
                    self.datapath.append((c,os.path.join(dirname,filename)))

        self.cache_size = cache_size # how many data points to cache in memory
        self.cache = {} # from index to (point_set, cls) tuple

        # if shuffle is None:
        #     if split == 'train': self.shuffle = True
        #     else: self.shuffle = False
        # else:
        #     

        self.reset()

    def _augment_batch_data(self, batch_data):
        noisy_data = provider.random_noise_point_cloud(batch_data)
        rotation_data = provider.rotation_point_cloud(noisy_data)
        shift_data = provider.shift_point_cloud(rotation_data)
        return shift_data


    def _get_item(self, index): 
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[fn[0]]
            cls = np.array([cls]).astype(np.int32)
            # point_set = np.loadtxt(fn[1],delimiter=',').astype(np.float32)
            # # Take the first npoints
            # point_set = point_set[0:self.npoints,:]
            # if self.normalize:
            #     point_set[:,0:3] = pc_normalize(point_set[:,0:3])
            # if not self.normal_channel:
            #     point_set = point_set[:,0:3]
            point_set = np.zeros((self.npoints, 3))
            df = pd.read_csv(fn[1])
            col_x = [name for i, name in enumerate(df.columns) if not i%2 and i > 1]
            col_y = [name for i, name in enumerate(df.columns) if i%2 and i > 1]
            num_sample = min(self.npoints, len(df.index))

            for i in range(num_sample): 
                kp = df.iloc[[i]]
                kps = []
                for icol in range(len(col_x)):
                    kps = np.append(kps, kp['id_frame'])
                    kps = np.append(kps, kp[col_x[icol]])
                    kps = np.append(kps, kp[col_y[icol]])
                point_set[i*len(col_x):(i+1)*len(col_x)] = np.reshape(kps, (-1, 3)).astype(np.float32)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)
        return point_set, cls
        
    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.datapath)

    # def num_channel(self):
    #     if self.normal_channel:
    #         return 6
    #     else:
    #         return 3

    def reset(self):
        self.idxs = np.arange(0, len(self.datapath))
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (len(self.datapath)+self.batch_size-1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, len(self.datapath))
        bsize = end_idx - start_idx
        # batch_data = np.zeros((bsize, self.npoints, self.num_channel()))
        batch_data = np.zeros((bsize, self.npoints, 3))
        batch_label = np.zeros((bsize), dtype=np.int32)
        for i in range(bsize):
            ps,cls = self._get_item(self.idxs[i+start_idx])
            batch_data[i] = ps
            batch_label[i] = cls
        self.batch_idx += 1
        if augment: batch_data = self._augment_batch_data(batch_data)
        return batch_data, batch_label