import os
import sys
sys.path.append(os.getenv("PO_ALGOS"))
from mnevergrad.mrecastlib import My_PYBOBYQA
import nevergrad as ng
import numpy as np
import time
from utils import retransform, transform, Func, run_funcs

def fun_mpybobyqa(obj_func, rand_init, in_dim, bounds_lower, bounds_upper, func_sleep, budget, seed, num_workers, inits, conn=None, timebound=None):
    start_time = time.time()
    np.random.seed(seed)

    database = []
    funCnt = 0
    algo = "mpybobyqa"
    # inits = []
    bounds_lower = np.array(bounds_lower)
    bounds_upper = np.array(bounds_upper)
    # rand_init = np.random.random(size=in_dim)
    # rand_init = np.ones(shape=in_dim)*0.5
    init_param = ng.p.Array(init=rand_init, lower=0, upper=1)
    # init_param = ng.p.Array(init=rand_init).set_bounds(0,1)
    stopFlag = False
    init_data = []

    while(not stopFlag and funCnt<budget):
        print("=========={} begin from {}:".format(algo, funCnt), flush=True)
        optimizers = []
        for _ in range(num_workers):
            seed = np.random.randint(2**32,dtype=np.uint32)
            init_param._set_random_state(np.random.RandomState(seed))
            optimizers.append(
                My_PYBOBYQA(parametrization=init_param,budget=budget-funCnt)
            )
        if not init_data:
            pass
        else:
            inits = [data[0] for data in init_data]
            initials_list = [
                    [(record['candidate_value'], record['value']) for record in data] 
                    for data in init_data
            ]
            # Reinitialize start
            for index, optimizer in enumerate(optimizers):
                optimizer.initials = initials_list[index]
        if not inits:
            for optimizer in optimizers:
                optimizer.initial_guess = rand_init
                rand_init = np.random.random(in_dim)
        else:
            # Always truly restart, just consider the use of inits
            for index, optimizer in enumerate(optimizers):
                record = inits[index]
                x0 = record['candidate_value']
                optimizer.initial_guess = x0

        
        while(not stopFlag and funCnt < budget):
            try:
                try:
                    candidate_chunk = [optimizer.ask() for optimizer in optimizers]
                except Exception as e:
                    print(repr(e), "self restart", flush=True)
                    rand_init = np.random.random(in_dim)
                    break
                x_chunk = [candidate.value for candidate in candidate_chunk]
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
                records = []
                for optimizer, candidate, x, x_real, value in zip(optimizers, candidate_chunk, x_chunk, x_real_chunk, fx_chunk):
                    optimizer.tell(candidate, value)
                    record_real = dict(number=funCnt, candidate_value=x_real, value=value, num_workers=num_workers, time=time.time()-start_time)
                    database.append(record_real) 
                    records.append(dict(candidate_value=x, value=value))
                funCnt += 1
                # print("algo {}, funCnt {}".format(algo, funCnt))
                if conn is not None:
                    conn.send(records)
                    # print("{} waiting for feedback".format(algo))
                    init_data, new_workers = conn.recv()
                    ## Restart
                    if init_data is not None:
                        num_workers = new_workers
                        stopFlag = False
                        break
                    if new_workers == 0:
                        stopFlag = True
                        break
                ## Normal stop
                if funCnt >= budget:
                    stopFlag = True
                delta_time = time.time() - start_time
                if (timebound is not None) and (delta_time > timebound): 
                    print("algo {} over timebound".format(algo), flush=True)
                    stopFlag = True
            except Exception as e:
                print(repr(e), flush=True)
                raise

    print("{} done".format(algo), flush=True)
    return database
    #print("{} done".format(algo))
    #print("database:", database)
    #recommendation = optimizer.recommend()
    #print(recommendation.value)
