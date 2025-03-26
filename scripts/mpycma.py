import cma
import numpy as np
import time
from utils import retransform, transform, Func, run_funcs

def fun_mpycma(obj_func, rand_init, in_dim, bounds_lower, bounds_upper, func_sleep, budget, seed, num_workers, inits, conn=None, timebound=None):
    start_time = time.time()
    np.random.seed(seed)

    database = []
    funCnt = 0
    algo = "mpycma"
    # inits = []
    bounds_lower = np.array(bounds_lower)
    bounds_upper = np.array(bounds_upper)
    # rand_init = np.random.random(size=in_dim)
    # rand_init = np.ones(shape=in_dim)*0.5
    stopFlag = False

    popsize = (int(4 + 3 * np.log(in_dim))//num_workers+1) * num_workers
    print("pycma popsize:", popsize)
    # maxiter = int(budget/popsize)
    # print("pycma maxiter:", maxiter)
    verbose = 3 #-9

    while(not stopFlag and funCnt<budget):
        print("=========={} begin from {}:".format(algo, funCnt))
        if not inits:
            x0 = rand_init
            # x0 = np.ones(shape=in_dim) * 0.5
        else:
            x0 = inits[0]['candidate_value']
        sigma0 = 0.1
        es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': popsize, 'bounds': [0, 1], 'seed': seed, 'verbose': verbose})
        #xopt, es = cma.fmin2(func, x0, sigma0, {'popsize': popsize, 'maxiter': maxiter, 'bounds': [0,1], 'seed': seed, 'verbose': verbose})

        ## Enter loop
        while(not stopFlag):
            ## Early stop
            if es.stop():
                print("pycma early stop")
                inits_real = sorted(database, key=lambda i:i['value'])
                init_candidate = retransform(inits_real[0]['candidate_value'], bounds_lower, bounds_upper)
                init_value = inits_real[0]['value']
                inits = [dict(candidate_value=init_candidate, value=init_value)]
                stopFlag = False
                tellFlag = False
            else:
                ## Normal loop
                #stopFlag = False
                tellFlag = True
                tell_list = []
                ask_num = 0
                while(ask_num < popsize):
                    ask_num += num_workers
                    x_chunk = es.ask(num_workers)
                    if funCnt == 0:
                        x_chunk[0] = rand_init
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
            if not tellFlag: # early stop or normal stop
                break # leave inner loop, then naturally enter outer loop
            else: # normal loop, tell
                X_tell = []
                fX_tell = []
                for x, value in sorted(tell_list, key=lambda i:i[1])[:popsize]:
                    X_tell.append(x)
                    fX_tell.append(value)
                es.tell(X_tell, fX_tell)

    print("{} done".format(algo), flush=True)
    return database
    #print("database:", database)
    #es.result_pretty()
