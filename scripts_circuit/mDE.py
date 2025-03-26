import nevergrad as ng
import numpy as np
import time
from utils import retransform, transform, Func, run_funcs

def fun_mDE(obj_func, rand_init, in_dim, bounds_lower, bounds_upper, func_sleep, budget, seed, num_workers, inits, conn=None, timebound=None):
    start_time = time.time()
    np.random.seed(seed)

    database = []
    totCnt = 0
    funCnt = 0
    algo = "DE"
    # inits = []
    bounds_lower = np.array(bounds_lower)
    bounds_upper = np.array(bounds_upper)
    update_bounds = True
    # rand_init = np.random.random(size=in_dim)
    # rand_init = np.ones(shape=in_dim)*0.5
    init_param = ng.p.Array(init=rand_init, lower=0, upper=1)
    # init_param = ng.p.Array(init=rand_init).set_bounds(0,1)
    stopFlag = False

    while(not stopFlag and funCnt<budget):
        print("=========={} begin from {}:".format(algo, funCnt))
        optimizer = ng.optimizers.TwoPointsDE(parametrization=init_param, budget=budget)
        if not inits:
            pass
        else:
            for record in inits[:optimizer.llambda]:
                restart_param = init_param.spawn_child()
                restart_param.value = record['candidate_value']
                restart_value = record['value']
                optimizer.tell(restart_param, restart_value)
            #optimizer.scale = float(1.0 / np.sqrt(in_dim))
            #optimizer._config.initialization = 'gaussian'
        #optimizer.register_callback("tell", update_database)

        ## Enter loop
        while(not stopFlag and funCnt < budget):
            candidate_chunk = [optimizer.ask() for _ in range(num_workers)]
            x_chunk = [candidate.value for candidate in candidate_chunk]
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
            for candidate, x, x_real, value in zip(candidate_chunk, x_chunk, x_real_chunk, fx_chunk):
                optimizer.tell(candidate, value)
                record_real = dict(number=funCnt, candidate_value=x_real, value=value, num_workers=num_workers, time=time.time()-start_time)
                database.append(record_real) 
                records.append(dict(candidate_value=x, value=value))
            funCnt += 1
            if conn is not None:
                conn.send(records)
                # print("{} waiting for feedback".format(algo))
                inits, new_workers = conn.recv()
                ## No restart, just update num_workers
                if inits is not None:
                    num_workers = new_workers
                    merge_param = init_param.spawn_child()
                    merge_param.value = inits[0]['candidate_value']
                    merge_value = inits[0]['value']
                    optimizer.tell(merge_param, merge_value)
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

    print("{} done".format(algo), flush=True)
    return database
    # print("database:", database)
    # recommendation = optimizer.recommend()
    # print(recommendation.value)
