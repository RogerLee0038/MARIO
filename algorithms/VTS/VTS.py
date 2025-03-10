# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 15:04:49 2022

@author: Zhao Aidong
"""

import numpy as np

from .TreeNode import TreeNode
from .utils import latin_hypercube, to_size
#from .Gibbs_slice_sampler import Gibbs_slice_sampling
from .LBFGS_torch import acq_min_msp
import torch

class VTS(object):
    #############################################

    def __init__(self, lb, ub, dims, ninits, init, with_init, func, iteration, Cp = 0, leaf_size = 20, kernel_type = "rbf", use_cuda = False, set_greedy=True):
        assert ninits <= leaf_size
        self.dims                    =  dims
        self.samples                 =  []
        self.nodes                   =  []
        self.Cp                      =  Cp
        self.sigma_x                 =  0.001
        
        self.lb                      =  lb
        self.ub                      =  ub
        self.ninits                  =  ninits
        self.init                    =  init
        self.with_init               =  with_init
        self.func                    =  func
        self.curt_best_value         =  float("inf")
        self.curt_best_sample        =  None
        self.sample_counter          =  0
        self.iterations              =  iteration
        
        self.LEAF_SIZE               =  leaf_size
        self.kernel_type             =  kernel_type
        
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.set_greedy = set_greedy
        
        self.init_train()
        
        root = TreeNode(self.samples, dims, max_leaf_size = leaf_size, parent = None, node_id = 0, kernel_type = kernel_type, device = self.device)
        self.nodes.append( root )
        self.node_counter = 1
        

    
        
    def split_node(self,nodeid):
        assert self.nodes[nodeid].num_samples >= self.LEAF_SIZE
        
        lchild_data, rchild_data = self.nodes[nodeid].split()
        lchildid, rchildid = self.node_counter, self.node_counter + 1
        
        lchild = TreeNode(lchild_data, self.dims, max_leaf_size = self.LEAF_SIZE, parent = nodeid, node_id = lchildid, kernel_type = self.kernel_type, device = self.device)
        self.nodes.append(lchild)
        
        rchild = TreeNode(rchild_data, self.dims, max_leaf_size = self.LEAF_SIZE, parent = nodeid, node_id = rchildid, kernel_type = self.kernel_type, device = self.device)
        self.nodes.append(rchild)
        
        self.node_counter += 2
        
        self.nodes[nodeid].update_child(lchildid,rchildid)

        
    def evaluate_funs(self, samples):
        #
        values = self.func(samples)
            
        for value, sample in zip(values,samples):
            if value < self.curt_best_value:
                self.curt_best_value  = value
                self.curt_best_sample = sample 
            self.sample_counter += 1
            dic = {'X':sample, 'Y':value}
            self.samples.append( dic )

    def evaluate_fun(self, sample):
        #
        value = self.func(sample)[0]
            
        if value < self.curt_best_value:
            self.curt_best_value  = value
            self.curt_best_sample = sample 
        self.sample_counter += 1
        dic = {'X':sample, 'Y':value}
        self.samples.append( dic )
        return dic
        
    def init_train(self):
        #latin hypercube sampling is used to generate init samples in search space
        if self.with_init:
            init_points = latin_hypercube(self.ninits-1, self.dims)
            init_points = to_size(init_points, self.lb, self.ub)
            init_points = np.vstack((self.init, init_points))
        else:
            init_points = latin_hypercube(self.ninits, self.dims)
            init_points = to_size(init_points, self.lb, self.ub)
        
        # for point in init_points:
        #     self.evaluate_fun(point)
        self.evaluate_funs(init_points)
            
        
        
    def update_recursive(self, samples, leaf_id):
        assert len(samples) > 0
        node_id = leaf_id
        while node_id is not None: 
            self.nodes[node_id].update(samples)
            node_id = self.nodes[node_id].parent
        

    def select(self):
        node_idx = 0
        path     = []
        nodelist = [0]
        
        while not self.nodes[node_idx].is_leaf():
            n_p = self.nodes[node_idx].num_samples
            UCT_lchild = self.nodes[self.nodes[node_idx].lchild].get_uct(n_p, self.set_greedy, self.Cp)
            UCT_rchild = self.nodes[self.nodes[node_idx].rchild].get_uct(n_p, self.set_greedy, self.Cp)
            #print('UCT_lchild = ',UCT_lchild,'UCT_rchild = ',UCT_rchild,'mu_lchild = ', self.nodes[self.nodes[node_idx].lchild].mu,'mu_rchild = ', self.nodes[self.nodes[node_idx].rchild].mu)
            if UCT_lchild >= UCT_rchild:
                path.append(0)  # 0 for lchild , 1 for rchild
                node_idx = self.nodes[node_idx].lchild
            else:
                path.append(1)  # 0 for lchild , 1 for rchild
                node_idx = self.nodes[node_idx].rchild
            nodelist.append(node_idx)
        self.nodes[node_idx].select_cell()
        print("Current node : ", node_idx )
        print("Path : ", path)
        return node_idx, path, nodelist
    
    
    def propose_samples_lcb(self, leaf_idx, path, nodelist, num_samples=20000):
        if len(path) > 0:
            #A = np.zeros((len(path),self.dims+1))
            A = np.array([self.nodes[j].split_hyperplane if k==0 else -self.nodes[j].split_hyperplane for j,k in zip(nodelist[:-1],path)])
            A = np.atleast_2d(A)
        GPs = self.nodes[leaf_idx].GPs
        cellidx = self.nodes[leaf_idx].selected_cell
        
        
        
        def lcb(Xin):
            x = np.atleast_2d(Xin)
            num = len(x)
            vals = 5*torch.ones((num,len(GPs))).double()
            in_region = np.bitwise_and(np.all(x>=self.lb,axis=1),np.all(x<=self.ub,axis=1))
            
            # x_1: num*(dim+1)  A: depth*(dim+1)    A * x_1: depth*num
            if len(path) > 0:
                in_region[np.any(A.dot(np.hstack((x,np.ones((num,1)))).T)>=0,axis=0)] = False
                #print(np.any(A.dot(np.hstack((x,np.ones((num,1)))).T)>0,axis=0))
                #print(in_region)
            
            # incell decision
            in_region[self.nodes[leaf_idx].kdtree.query(x, eps=0, k=1)[1]!=cellidx] = False
            
            vals[in_region] = torch.atleast_2d( GPs[0].LCB_nograd(torch.tensor(x[in_region],device=self.device)).detach()).T
            #print('lcb_vals = ',vals.T[0])
            return vals.T[0]
        
        cyc = 10
        num_sample = num_samples//cyc
        x0 = self.nodes[leaf_idx].X[cellidx]
        r = 1.0 * np.std(self.nodes[leaf_idx].X,axis = 0) + self.sigma_x
        lcb_min = np.inf
        
        for j in range(cyc):
            target_region = np.array([np.maximum(x0-r,self.lb), np.minimum(x0+r,self.ub)])
            
            x_sample = to_size(latin_hypercube(num_sample, self.dims), target_region[0], target_region[1])
            lcb_val = 5 * np.ones(num_sample)
            
            incell = np.bitwise_and(np.all(x_sample>=self.lb,axis=1),np.all(x_sample<=self.ub,axis=1))
            if len(path) > 0:
                incell[np.any(A.dot(np.hstack((x_sample,np.ones((num_sample,1)))).T)>=0,axis=0)] = False
            
            incell[self.nodes[leaf_idx].kdtree.query(x_sample, eps=0, k=1)[1]!=cellidx] = False
            #incell = self.kdtree.query(x_sample, eps=0, k=1)[1] == self.target_cell
            lcb_val[incell] = GPs[0].LCB_nograd(torch.tensor(x_sample[incell],device=self.device)).detach().numpy()
            #if lcb_val.min() < 0 and lcb_val.min() < lcb_min:

            # if lcb_val.min() < 0:
            #     x_init = x_sample[np.argmin(lcb_val)]
            #     print('lcb_init = ',lcb_val.min())
            #     break
            # else:
            #     r = 0.8 * r

            lcb_cur_min = np.min(lcb_val)
            if lcb_val.min() < lcb_min:
                x_init = x_sample[np.argmin(lcb_val)]
                lcb_min = lcb_cur_min
            if lcb_min < 0:
                print('lcb_init = ',lcb_val.min())
                break
            else:
                r = 0.8 * r
        
        #print(target_region)
        proposed_X, acq_value = acq_min_msp(lambda x:lcb(x), lambda x:finite_diff(x,lcb), torch.tensor(x_init,device = self.device), torch.tensor(target_region,device = self.device), n_warmup=10000)
        #Gibbs_slice_sampling(lcb, x0, np.array([self.lb,self.ub]), f_min=-5)
        print('lcb_val = ',acq_value)
        return proposed_X
    
    
    
    def propose_samples_ei(self, leaf_idx, path, nodelist, num_samples=20000):
        if len(path) > 0:
            #A = np.zeros((len(path),self.dims+1))
            A = np.array([self.nodes[j].split_hyperplane if k==0 else -self.nodes[j].split_hyperplane for j,k in zip(nodelist[:-1],path)])
            A = np.atleast_2d(A)
        GPs = self.nodes[leaf_idx].GPs
        cellidx = self.nodes[leaf_idx].selected_cell
        
        
        
        def ei(Xin):
            x = np.atleast_2d(Xin)
            num = len(x)
            vals = -5*torch.ones((num,len(GPs))).double()
            in_region = np.bitwise_and(np.all(x>=self.lb,axis=1),np.all(x<=self.ub,axis=1))
            
            # x_1: num*(dim+1)  A: depth*(dim+1)    A * x_1: depth*num
            if len(path) > 0:
                in_region[np.any(A.dot(np.hstack((x,np.ones((num,1)))).T)>=0,axis=0)] = False
                #print(np.any(A.dot(np.hstack((x,np.ones((num,1)))).T)>0,axis=0))
                #print(in_region)
            
            # incell decision
            in_region[self.nodes[leaf_idx].kdtree.query(x, eps=0, k=1)[1]!=cellidx] = False
            
            vals[in_region] = torch.atleast_2d( GPs[0].EI_nograd(torch.tensor(x[in_region],device=self.device)).detach()).T
            #print('lcb_vals = ',vals.T[0])
            return vals.T[0]
        
        cyc = 16
        num_sample = num_samples//cyc
        x0 = self.nodes[leaf_idx].X[cellidx]
        r = 1.0 * np.std(self.nodes[leaf_idx].X,axis = 0) + self.sigma_x
        
        for j in range(cyc):
            target_region = np.array([np.maximum(x0-r,self.lb), np.minimum(x0+r,self.ub)])
            
            x_sample = to_size(latin_hypercube(num_sample, self.dims), target_region[0], target_region[1])
            ei_val = -5 * np.ones(num_sample)
            
            incell = np.bitwise_and(np.all(x_sample>=self.lb,axis=1),np.all(x_sample<=self.ub,axis=1))
            if len(path) > 0:
                incell[np.any(A.dot(np.hstack((x_sample,np.ones((num_sample,1)))).T)>=0,axis=0)] = False
            
            incell[self.nodes[leaf_idx].kdtree.query(x_sample, eps=0, k=1)[1]!=cellidx] = False
            #incell = self.kdtree.query(x_sample, eps=0, k=1)[1] == self.target_cell
            ei_val[incell] = GPs[0].EI_nograd(torch.tensor(x_sample[incell],device=self.device)).detach().numpy()
            #if lcb_val.min() < 0 and lcb_val.min() < lcb_min:
            if ei_val.max() > 0:
                x_init = x_sample[np.argmax(ei_val)]
                print('ei_init = ',ei_val.max())
                break
            else:
                r = 0.7 * r
        
        #print(target_region)
        proposed_X, acq_value = acq_min_msp(lambda x:-ei(x), lambda x:-finite_diff(x,ei), torch.tensor(x_init,device = self.device), torch.tensor(target_region,device = self.device), n_warmup=10000)
        #Gibbs_slice_sampling(lcb, x0, np.array([self.lb,self.ub]), f_min=-5)
        print('ei_val = ',-acq_value)
        return proposed_X
    
    
    '''
    def propose_samples_ei(self, leaf_idx, path, nodelist):
        if len(path) > 0:
            #A = np.zeros((len(path),self.dims+1))
            A = np.array([self.nodes[j].split_hyperplane if k==0 else -self.nodes[j].split_hyperplane for j,k in zip(nodelist[:-1],path)])
            A = np.atleast_2d(A)
        GPs = self.nodes[leaf_idx].GPs
        cellidx = self.nodes[leaf_idx].selected_cell
        def ei(Xin):
            x = np.atleast_2d(Xin)
            num = len(x)
            vals = 0*np.ones((num,len(GPs)))
            in_region = np.bitwise_and(np.all(x>=self.lb,axis=1),np.all(x<=self.ub,axis=1))
            
            # x_1: num*(dim+1)  A: depth*(dim+1)    A * x_1: depth*num
            if len(path) > 0:
                in_region[np.any(A.dot(np.hstack((x,np.ones((num,1)))).T)>0,axis=0)] = False
                #print(np.any(A.dot(np.hstack((x,np.ones((num,1)))).T)>0,axis=0))
                #print(in_region)
            
            # incell decision
            in_region[self.nodes[leaf_idx].kdtree.query(x, eps=0, k=1)[1]!=cellidx] = False
            
            vals[in_region] = np.maximum(0,np.atleast_2d( GPs[0].EI_nograd(torch.tensor(x[in_region],device=self.device)).detach().numpy()).T)
            
            return vals
        x0 = self.nodes[leaf_idx].X[cellidx]
        proposed_X, acq_value = Gibbs_slice_sampling(ei, x0, np.array([self.lb,self.ub]), f_min=0)
        return proposed_X
    '''
    

    def search(self):
        for idx in range(self.sample_counter, self.iterations):
            print("")
            print("#"*20)
            print("Iteration:", idx)
            
            leaf_idx, path, nodelist = self.select()
            
            #xsample = self.propose_samples_ei( leaf_idx, path, nodelist )
            xsample = self.propose_samples_lcb( leaf_idx, path, nodelist )
            print('xsample = ',xsample)
            samples = [self.evaluate_fun( xsample)]
            print("samples:", samples)
            #update
            self.update_recursive(samples, leaf_idx)
            #split
            if self.nodes[leaf_idx].is_splittable():
                self.split_node(leaf_idx)
            
            print("Total samples:", len(self.samples) )
            print("Current best f(x):", self.curt_best_value )
            # print("current best x:", np.around(self.curt_best_sample, decimals=1) )
            #print("Current best x:", self.curt_best_sample )



def finite_diff(x_tensor,f,epslong=1e-8):
    with torch.no_grad():
        dims = len(x_tensor)
        delta = epslong*torch.eye(dims,device = x_tensor.device)
        ys = f(torch.cat((x_tensor + delta,x_tensor - delta),dim = 0))
        grad = (ys[:dims] - ys[dims:])/(2*epslong)
    return grad
