# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 18:03:49 2022

@author: Zhao Aidong
"""

import numpy as np


def latin_hypercube(n, dims):
    points = np.zeros((n, dims))
    centers = np.arange(n)
    centers = centers / float(n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(np.arange(n))]

    perturbation = np.random.rand(n, dims) 
    perturbation = perturbation / float(n)
    points += perturbation
    return points

def to_size(x, lb, ub):
    return lb + (ub-lb) * x


def get_conditional_dist(joint_mu, joint_cov, var_idx):
    '''Returns the conditional distribution given the joint distribution and which variable
    the conditional probability should use.
    
      joint_mu: joint distribution's mu
      joint_cov: joint distribution's covariance
      var_index: index of the variable in the joint distribution. Everything else will be 
        conditioned on. 
      
    returns:
      a function that can sample from the univariate conditional distribution
    '''
    var_index = np.arange(len(joint_mu))==var_idx
    a = joint_mu[var_index]
    b = joint_mu[~var_index]
    
    A = joint_cov[var_index, var_index]
    #B = joint_cov[~var_index, ~var_index]
    B = joint_cov[~var_index][:, ~var_index]
    C = joint_cov[var_index][:, ~var_index].reshape(-1)
    
    # we're dealing with one dimension so
    B_inv = np.linalg.inv(B)
    
    # Return a function that can sample given a value of g
    def dist(x, size = 1):
        g = x[~var_index]
        # a + C*B^{-1}(g - b)
        mu = a + C.dot( B_inv ).dot (g - b)
        # A - C * B^{-1} * C^T
        
        cov = A - C.dot(B_inv).dot(C)
        
        return np.sqrt(cov) * np.random.randn(size) + mu
    
    return dist


def Gibbs_slice_sampling(f, x0, f_min = 0, sample_iter = 1000, batch_size = 20, device = 'cpu'):
    dim = len(x0)
    mu = x0
    C = 1.0*np.eye(dim)
    x_init = x0 + np.random.multivariate_normal(np.zeros(dim),0.1*np.eye(dim),batch_size)
    
    num = 10
    fx = f(x_init)
        
    # record optimum
    idx = fx.argmax()
    xopt = x_init[idx]
    yopt = fx[idx]
    
    
    y = np.repeat(fx,num)

    
    idxAccept =  np.arange(batch_size*num)
    cut_min = f_min * np.ones(batch_size*num)
    samples = np.array([])
    for k in range(sample_iter):
        cut_min = 0.2*f_min + 0.8*y
        yl = cut_min[idxAccept]
        yu = y[idxAccept]
         
        y[idxAccept] = yl + (yu - yl) * np.random.rand(len(idxAccept))
        
        # sampling x
        var_idx = k % dim
        dist = get_conditional_dist(mu, C, var_idx)
        
        x_var_idx = np.array([dist(s,num) for s in x_init]).reshape(-1)
        x_sample = np.repeat(np.atleast_2d(x_init),num,axis=0) 
        #cut_min = np.repeat(cut_min,num) 
        #y = np.repeat(y,num) 
        
        x_sample[:,var_idx] = x_var_idx
        
        # get f(x)
        fx=f(x_sample)             
        
        #update optima
        idx = fx.argmax()
        if fx[idx] > yopt:
            xopt = x_sample[idx]
            yopt = fx[idx]
        
        
        idxAccept=[idx for idx,val in enumerate(fx) if val>cut_min[idx] and val>y[idx]]
        
       
        acc_samples = x_sample[idxAccept,:]
        if len(samples)==0:
            samples = acc_samples
        else:
            samples = np.vstack((samples, np.atleast_2d(acc_samples)))
            
        newx_idx = fx.reshape((batch_size,num)).argmax(axis=1)
        x_init = np.array([x_sample[idx*num + xs] for idx,xs in enumerate(newx_idx)])
        
        # update
        c = 0.7
        mu = c * mu + (1-c) * np.mean(x_sample[idxAccept],axis=0)
        C = c * C + (1-c) * np.cov(x_sample[idxAccept].T)

    return xopt,yopt




def Ackley(xin):
    xs = np.atleast_2d(xin)*15-5
    result = np.array([(-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e ) for x in xs])
    print('val = ',result)
    return -result


if __name__ == "__main__":
    x0 = 0.6*np.ones(10)
    x,y = Gibbs_slice_sampling(Ackley, x0, f_min = -20, sample_iter = 50, batch_size = 20, device = 'cpu')



