import os
import sys
sys.path.append(os.getenv("PO_ALGOS"))
from TuRBO import TurboM
import torch
import numpy as np
import time
from utils import retransform, transform, Func, run_funcs

def fun_mturbo(obj_func, rand_init, in_dim, bounds_lower, bounds_upper, func_sleep, budget, seed, num_workers, inits, conn=None, timebound=None):
    assert conn is None # just for compare
    start_time = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    database = []
    funCnt = 0
    algo = "mturbo"
    # inits = []
    bounds_lower = np.array(bounds_lower)
    bounds_upper = np.array(bounds_upper)
    # rand_init = np.random.random(size=in_dim)
    # rand_init = np.ones(shape=in_dim) * 0.5
    stopFlag = False

    batch_size = (int(4 + 3 * np.log(in_dim))//num_workers+1) * num_workers
    print("turbo batch_size:", batch_size)

    def func(x_2d: np.ndarray) -> np.ndarray:
        nonlocal obj_func, budget, func_sleep, num_workers, funCnt, database, start_time
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

            ## Normal stop
            if funCnt >= budget:
                raise Exception('out of budget')
            delta_time = time.time() - start_time
            if (timebound is not None) and (delta_time > timebound): 
                print("algo {} over timebound".format(algo), flush=True)
                raise Exception('out of time')

        return np.array(fX_list)[:,np.newaxis]

    while(not stopFlag and funCnt<budget):
        print("=========={} begin from {}:".format(algo, funCnt))
        if not inits:
            x0 = rand_init
        else:
            x0 = inits[0]['candidate_value']
        turboM = TurboM(
            f=func,
            lb=np.zeros(in_dim),
            ub=np.ones(in_dim),
            n_init=max(2*in_dim//num_workers, 1)*num_workers,
            max_evals=num_workers*budget,
            n_trust_regions=1,
            init=x0[np.newaxis,:],
            with_init=True,
            batch_size=batch_size,
            verbose=True,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=50,
            min_cuda=1024,
            device="cpu",
            dtype="float64",
        )
        ## Enter loop
        try:
            turboM.optimize()  # Run optimization
        except Exception as e:
            print(repr(e))
            delta_time = time.time() - start_time
            if (timebound is not None) and (delta_time > timebound): 
                stopFlag = True
            elif funCnt >= budget:
                stopFlag = True
            else:
                raise

    return database
    # print("{} done".format(algo))
    # print("database:", database)
    # print(turboM.X, turboM.fX)
