import os
import sys
sys.path.append(os.getenv("PO_ALGOS"))
from TuRBO import TurboAt
import torch
import numpy as np
import time
from utils import retransform, transform, Func, run_funcs

def fun_mturbo_at(obj_func, rand_init, in_dim, bounds_lower, bounds_upper, func_sleep, budget, seed, num_workers, inits, conn=None, timebound=None):
    start_time = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    database = []
    totCnt = 0
    funCnt = 0
    algo = "mturbo_at"
    # inits = []
    bounds_lower = np.array(bounds_lower)
    bounds_upper = np.array(bounds_upper)
    # rand_init = np.random.random(size=in_dim)
    # rand_init = np.ones(shape=in_dim) * 0.5
    stopFlag = False

    batch_size = (int(4 + 3 * np.log(in_dim))//num_workers+1) * num_workers
    print("turbo_at batch_size:", batch_size)
    n_init = max(2*in_dim//num_workers, 1) * num_workers

    while(not stopFlag and funCnt<budget):
        print("=========={} begin from {}:".format(algo, funCnt))
        if not inits:
            x0 = rand_init
        else:
            x0 = inits[0]['candidate_value']
        turbo_at = TurboAt(
            lb=np.zeros(in_dim),
            ub=np.ones(in_dim),
            n_init=n_init,
            max_evals=np.inf,
            init = x0[np.newaxis,:],
            with_init = True,
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
        while(not stopFlag):
            ## Early stop
            if turbo_at.stop():
                print("turbo_at early stop")
                turbo_at._restart()
                #inits_real = sorted(database, key=lambda i:i['value'])
                #init_candidate = retransform(inits_real[0]['candidate_value'], bounds_lower, bounds_upper)
                #init_value = inits_real[0]['value']
                #inits = [dict(candidate_value=init_candidate, value=init_value)]
                stopFlag = False
                tellFlag = False
            else:
                ## Normal loop
                #stopFlag = False
                tellFlag = True
                tell_list = []
                ask_num = 0
                batch_size = (int(4 + 3 * np.log(in_dim))//num_workers+1) * num_workers
                turbo_at.batch_size = batch_size
                target_num = n_init if turbo_at.is_random() else batch_size
                while(ask_num < target_num):
                    try:
                        ask_num += num_workers
                        x_chunk = turbo_at.ask(num_workers)
                        x_real_chunk = [transform(x, bounds_lower, bounds_upper) for x in x_chunk]
                        if num_workers > 1:
                            fx_chunk, totCnt = run_funcs(
                                Func(obj_func, func_sleep),
                                x_real_chunk,
                                num_workers,
                                totCnt
                            )
                        else:
                            x = x_real_chunk[0]
                            fx_chunk = [Func(obj_func, func_sleep)(x, totCnt)]
                            totCnt += 1
                        records = []
                        for x, x_real, value in zip(x_chunk, x_real_chunk, fx_chunk):
                            tell_list.append((x, value))
                            record_real = dict(number=funCnt, candidate_value=x_real, value=value, num_workers=num_workers, time=time.time()-start_time)
                            database.append(record_real) 
                            records.append(dict(candidate_value=x, value=value))
                        funCnt += 1
                        # print("algo {}, funCnt {}".format(algo, funCnt))
                        if conn is not None:
                            conn.send(records)
                            # print("{} waiting for feedback".format(algo))
                            inits, new_workers = conn.recv()
                            ## No restart, just update num_workers
                            if inits is not None:
                                num_workers = new_workers
                                merge_param_value = inits[0]['candidate_value']
                                merge_value = inits[0]['value']
                                if not any(
                                [np.all(merge_param_value==x) 
                                    for x,_ in tell_list]
                                        ):
                                    tell_list.append((merge_param_value, merge_value))
                            ## Terminate
                            if new_workers == 0:
                                stopFlag = True
                                tellFlag = False
                                break
                        ## Normal stop
                        if funCnt >= budget:
                            stopFlag = True
                            tellFlag = False
                            break
                        delta_time = time.time() - start_time
                        if (timebound is not None) and (delta_time > timebound): 
                            print("algo {} over timebound".format(algo), flush=True)
                            stopFlag = True
                            tellFlag = False
                            break
                    except Exception as e:
                        print(repr(e), flush=True)
                        raise
            if stopFlag: # terminate or normal stop
                break # leave inner loop, then naturally enter outer loop
            elif (not stopFlag) and (not tellFlag): # _restart()
                continue
            else: # normal loop, tell
                X_tell = []
                fX_tell = []
                for x, value in tell_list:
                    X_tell.append(x)
                    fX_tell.append(value)
                turbo_at.tell(np.array(X_tell), np.array(fX_tell)[:, np.newaxis])

    print("{} done".format(algo), flush=True)
    return database
    # print("database:", database)
    # print(turboM.X, turboM.fX)
