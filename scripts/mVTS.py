import os
import sys
sys.path.append(os.getenv("PO_ALGOS"))
from VTS import VTS
import torch
import numpy as np
import time
from utils import retransform, transform, Func, run_funcs
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def fun_mVTS(obj_func, rand_init, in_dim, bounds_lower, bounds_upper, func_sleep, budget, seed, num_workers, inits, conn=None, timebound=None):
    assert conn is None # just for compare
    assert num_workers == 1 # serial limited
    start_time = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    database = []
    funCnt = 0
    algo = "mVTS"
    # inits = []
    bounds_lower = np.array(bounds_lower)
    bounds_upper = np.array(bounds_upper)
    # rand_init = np.random.random(size=in_dim)
    # rand_init = np.ones(shape=in_dim) * 0.5
    stopFlag = False

    def func(xin: np.ndarray) -> np.ndarray:
        nonlocal obj_func, budget, func_sleep, num_workers, funCnt, database, start_time, budget
        x_2d = np.atleast_2d(xin)
        fX_list = []
        X_chunks = [x_2d[i:i+num_workers] for i in range(0,len(x_2d),num_workers)]
        for x_chunk in X_chunks:
            x_real_chunk = [transform(x, bounds_lower, bounds_upper) for x in x_chunk]
            if num_workers > 1:
                fx_chunk = run_funcs(
                    Func(obj_func, func_sleep),
                    x_real_chunk,
                    num_workers,
                )
            else:
                x = x_real_chunk[0]
                fx_chunk = [Func(obj_func, func_sleep)(x)]
            for x_real, value in zip(x_real_chunk, fx_chunk):
                record_real = dict(number=funCnt, candidate_value=x_real, value=value, num_workers=num_workers, time=time.time()-start_time)
                database.append(record_real)
                fX_list.append(value)
            funCnt += 1
            assert funCnt < budget
            delta_time = time.time() - start_time
            if (timebound is not None) and (delta_time > timebound): 
                print("algo {} over timebound".format(algo), flush=True)
                raise Exception('out of time')
        return np.array(fX_list)[:,np.newaxis]

    print("=========={} begin from {}:".format(algo, funCnt))
    if not inits:
        x0 = rand_init
    else:
        x0 = inits[0]['candidate_value']
    optimizer = VTS(
                 np.zeros(in_dim),              
                 # the lower bound of each problem dimensions
                 np.ones(in_dim),              
                 # the upper bound of each problem dimensions
                 in_dim,          
                 # the problem dimensions
                 ninits = 2*in_dim//num_workers*num_workers,      
                 # the number of random samples used in initializations 
                 init = x0,
                 with_init = True,
                 func = func,               
                 # function object to be optimized
                 iteration = budget,    
                 # maximal iteration to evaluate f(x)
                 Cp = 0,              
                 # Cp for UCT function
                 leaf_size = 4*in_dim,  # tree leaf size
                 kernel_type = 'rbf',   # kernel for GP model
                 use_cuda = False,     #train GP on GPU
                 set_greedy = True    # greedy setting
                 )
    ## Enter loop
    try:
        optimizer.search()  # Run optimization
    except Exception as e:
        print(repr(e))
        delta_time = time.time() - start_time
        if (timebound is not None) and (delta_time > timebound): 
            print("VTS done")
        elif funCnt >= budget:
            print("VTS done")
        else:
            raise

    return database
