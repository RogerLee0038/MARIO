# 
from .GP import train_gp
import numpy as np
from scipy.spatial import KDTree

class TreeNode(object):
    
    #X:num_samples*dims  Y:num_samples*(1+num_constrains)
    def __init__(self, samples, dims, max_leaf_size = 20, parent = None, node_id = 0, kernel_type = "rbf", device = 'cpu'):
        
        self.dims          = dims
        
        self.num_samples   = len(samples)
            
        self.max_leaf_size = max_leaf_size
        self.gp_kernel     = kernel_type
        
        self.parent        = parent        
        self.lchild        = None
        self.rchild        = None
        self.split_hyperplane = None
        
        self.X             = np.array([s['X'] for s in samples])
        self.Y             = np.array([s['Y'] for s in samples])
        self.device        = device
        self.kdtree        = KDTree(self.X)  
        
        self.mu            = np.mean(self.Y,axis=0)
        self.var           = np.var(self.Y,axis=0)

        self.id            = node_id
        self.optima        = np.min(self.Y)
                
    def mean_std(self):
        return self.mu, np.sqrt(self.var)
    
    def is_splittable(self):
        return self.num_samples >= self.max_leaf_size
        
    def is_root(self):
        return self.parent == None
        
    def is_leaf(self):
        return self.lchild == None and self.rchild == None
    
    def Q_value(self):
        #assert not self.is_leaf()
        return -self.optima #-self.mu[0] #
    
    def update(self, samples):
        assert len(samples) > 0
        for newsample in samples:
            if self.is_leaf():
                if self.num_samples == 0:
                    self.X = np.atleast_2d(newsample['X'])
                    self.Y = np.atleast_2d(newsample['Y'])
                else:
                    self.X = np.vstack((self.X, np.atleast_2d(newsample['X'])))
                    self.Y = np.vstack((self.Y, np.atleast_2d(newsample['Y'])))
                self.kdtree = KDTree(self.X) 
                
            if newsample['Y'][0]<self.optima:
                self.optima = newsample['Y'][0]
            self.num_samples += 1
            mu_bk = self.mu
            c1,c2 = 1-1./self.num_samples, 1./self.num_samples
            self.mu = c1 * mu_bk + c2 * newsample['Y']
            self.var = c1 * self.var + c2 * (newsample['Y'] - mu_bk) * (newsample['Y'] - self.mu)
        if self.is_leaf():
            self.kdtree = KDTree(self.X)
        
    
    
    def select_cell(self):
        assert self.is_leaf()
        #s = 0
        self.selected_cell = np.argmin(self.Y)
        print('target cell = ',self.selected_cell,'y = ',self.Y[ self.selected_cell])
        self.GPs = train_gp(self.X,self.Y,use_ard=False, num_steps=200,bounds = np.array([[0]*self.dims,[1]*self.dims]))
        return self.X[self.selected_cell,:]

    def get_uct(self, n_p, set_greedy = True, Cp = 10 ):
        if self.parent == None:
            return float('inf')
        if self.num_samples == 0:
            return float('inf')
        #return np.exp(self.Q_value()) / self.num_samples + Cp*np.sqrt( np.log2(n_p) / self.num_samples )
        if set_greedy:
            return self.Q_value()
        else:
            return np.exp(self.Q_value()) / self.num_samples + Cp*np.sqrt( np.log2(n_p) / self.num_samples )
    
        
    def split(self):
        assert self.num_samples >= 2
        centroids = kMeans(self.X, self.Y)
        A = centroids[0]-centroids[1]
        A = A/np.linalg.norm(A,2)
        b = np.dot(A,(centroids[0]+centroids[1])/2)
        self.split_hyperplane = np.append(A,-b)
        is_lchild = np.hstack((self.X,np.ones((self.num_samples,1)))).dot(self.split_hyperplane) <= 0
        lchild_data = [{'X': s, 'Y': t} for s,t in zip(self.X[is_lchild],self.Y[is_lchild])]
        rchild_data = [{'X': s, 'Y': t} for s,t in zip(self.X[np.bitwise_not(is_lchild)],self.Y[np.bitwise_not(is_lchild)])]
        
        del(self.X,self.Y,self.kdtree,self.GPs,self.device)
        assert len( lchild_data ) + len( rchild_data ) ==  self.num_samples
        assert len( lchild_data ) > 0 
        assert len( rchild_data ) > 0 
        return lchild_data, rchild_data
    
    def update_child(self,lchild,rchild):
        self.lchild = lchild
        self.rchild = rchild
        return






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
    # print(np.array([X[idx0],X[idx1]]))
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


