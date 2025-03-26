import numpy as np

def kMeans(X, Y, k = 2, max_iter = 20):
    w = 0.7
    npnts, dims = X.shape
    n_funs = Y.shape[1]
    centroids = np.zeros((k,dims))
    Y_centroids = np.zeros((k,n_funs))
    x_weights = np.std(X,axis=0)
    weights = np.std(Y,axis=0)
    
    weighted_dists = np.zeros((npnts,npnts))
    dists = np.zeros((npnts,npnts))
    f_dists = np.zeros((npnts,npnts))
    for j in range(npnts):
        dists[j] = np.linalg.norm((X-X[j,:])/x_weights, ord=2, axis=1) 
        f_dists[j] = np.linalg.norm((Y-Y[j,:])/weights, ord=2, axis=1)
        
    weighted_dists = w * dists/np.sqrt(dims) + (1-w) * f_dists/np.sqrt(n_funs)
    idxes = weighted_dists.argmax()
    idx0 = idxes//npnts
    idx1 = idxes%npnts
    print("in TreeNode kMeans for centroids\n", np.array([X[idx0],X[idx1]]))
    centroids[:2,:]=np.array([X[idx0],X[idx1]])
    Y_centroids[:2,:] = np.array([Y[idx0],Y[idx1]])
    
    for j in range(k-2):
        weighted_dists = np.zeros((j+1,npnts))
        for i in range(j+1):
            dists[i] = w * np.linalg.norm((X-centroids[i,:])/x_weights, ord=2, axis=1)/np.sqrt(dims) + (1-w) * np.linalg.norm((Y-Y_centroids[i,:])/weights, ord=2, axis=1)/np.sqrt(n_funs)
        minval, _ = dists.min(axis=0)
        centroids[j+1] = X[minval.argmax()]
        Y_centroids[j+1] = Y[minval.argmax()]
        
    weighted_dists = np.zeros((k,npnts))
    for i in range(max_iter):
        centroids_bk = np.array(centroids)
        for j in range(k):
            weighted_dists[j] = w * np.linalg.norm((X-centroids[j,:])/x_weights, ord=2, axis=1)/np.sqrt(dims) + (1-w) * np.linalg.norm((Y-Y_centroids[j,:])/weights, ord=2, axis=1)/np.sqrt(n_funs)
        clusters = weighted_dists.argmin(axis = 0)
        
        for j in range(k):
            centroids[j] = np.mean(X[clusters==j],axis=0)
        max_variation = abs(centroids_bk - centroids).max()
        if max_variation<=1e-4:
            break

    return centroids

class TreeNode(object):
    #X:num_samples*dims  Y:num_samples*(1+num_constrains)
    def __init__(self, data, dims, max_leaf_size = 20, no_split = False):
        self.dims          = dims
        self.num_data   = len(data)
        self.max_leaf_size = max_leaf_size
        self.lchild        = None
        self.rchild        = None
        self.split_hyperplane = None
        self.data          = sorted(data, key=lambda data:data['value'])
        self.best_data     = self.data[0]
        self.X             = np.vstack([r['candidate_value'] for r in data])
        self.Y             = np.vstack([r['value'] for r in data])
        self.rmax = np.max(np.max(self.X, axis=0) - np.min(self.X, axis=0))
        self.no_split = no_split

    def add_data(self, record):
        if record['value'] < self.data[0]['value']:
            self.data.insert(0, record)
            self.best_data = record
        else:
            self.data.append(record)
    
    def is_splittable(self):
        if self.no_split:
            return False
        else:
            return self.rmax > 0.01 and self.num_data >= self.max_leaf_size
        
    def get_best(self):
        return self.best_data
    
    def split(self):
        assert self.num_data >= 2, "data less than 2"
        centroids = kMeans(self.X, self.Y)
        A = centroids[0]-centroids[1]
        A = A/np.linalg.norm(A,2)
        b = np.dot(A,(centroids[0]+centroids[1])/2)
        self.split_hyperplane = np.append(A,-b)
        is_lchild = np.hstack((self.X,np.ones((self.num_data,1)))).dot(self.split_hyperplane) <= 0
        lchild_data = [{'candidate_value': s, 'value': t[0]} for s,t in zip(self.X[is_lchild],self.Y[is_lchild])]
        rchild_data = [{'candidate_value': s, 'value': t[0]} for s,t in zip(self.X[np.bitwise_not(is_lchild)],self.Y[np.bitwise_not(is_lchild)])]
        
        assert len( lchild_data ) + len( rchild_data ) ==  self.num_data, "sum of child data does not equal to the parent's"
        #assert len( lchild_data ) > 0, "left child data less than 1" 
        #assert len( rchild_data ) > 0, "right child data less than 1" 
        return lchild_data, rchild_data
    
    def provide_data(self):
        assert not self.is_splittable(), "provide data just for leaf nodes"
        temp_data = [data for data in self.data 
                     if np.max(
                        np.abs(data['candidate_value'] - self.best_data['candidate_value'])
                        ) < 0.4
                    ]
        if len(temp_data) > self.max_leaf_size:
            clip_data = [self.best_data]
            for i in np.random.choice(len(temp_data)-1, self.max_leaf_size):
                clip_data.append(temp_data[i+1])
            return clip_data
        else:
            return temp_data
