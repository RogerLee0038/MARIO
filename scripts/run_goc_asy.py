import sys
import os
sys.path.append(os.getenv("PO_BENCHS"))
from go_functions import my_funcs
import multiprocessing as mp
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
# import multiprocessing.pool
import numpy as np
from mbobyqa import fun_mbobyqa
from mlbfgsb import fun_mlbfgsb
from mpycma import fun_mpycma
from mDE import fun_mDE
from mABBO import fun_mABBO
from mShiwa import fun_mShiwa
from mturbo import fun_mturbo
from mturbo_at import fun_mturbo_at
from mOnePlusOne import fun_mOnePlusOne
from mPSO import fun_mPSO
from mRandomSearch import fun_mRandomSearch
from mTBPSA import fun_mTBPSA
from mneldermead import fun_mneldermead
from mpyneldermead import fun_mpyneldermead
from mDIRECT import fun_mDIRECT
from mVTS import fun_mVTS
from mpyVTS import fun_mpyVTS
from mpybobyqa import fun_mpybobyqa
from feedback_asy import fun_feedback
from utils import transform, retransform
import tomli
import pickle
import time

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, val):
        pass

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonProcessPool(mp.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc

def process_run(func, args):
    result = func(*args)
    print("{} says that {} done".format(
        mp.current_process().name,
        func.__name__)
    )
    return result

def process_runstar(args):
    return process_run(*args)

if __name__ == '__main__':
    start_time = time.time()
    with open("conf.toml", "rb") as f:
        toml_dict = tomli.load(f)
    algo_list = toml_dict["algorithm"]["algo_list"]
    num_worker_list = toml_dict["algorithm"]["num_worker_list"]
    together = toml_dict["algorithm"]["together"]
    seed = toml_dict["algorithm"]["seed"]
    func_name = toml_dict['target']["func_name"]
    #in_dim = toml_dict['target']["in_dim"]
    func_sleep = toml_dict['target']["func_sleep"]
    #rotation = toml_dict['target']["rotation"]
    #n_blocks = toml_dict['target']["n_blocks"]
    #uv_factor = toml_dict['target']["uv_factor"]
    in_dim = toml_dict['target'].get('in_dim')
    inits_file = toml_dict['target'].get('inits_file')
    if inits_file is not None:
        with open(inits_file, "rb") as f:
            inits_real = pickle.load(f)
    else:
        inits_real = []

    budget_factor = toml_dict['strategy']["budget_factor"]
    pass_percent = toml_dict['strategy']["pass_percent"]
    pass_divide = toml_dict['strategy']["pass_divide"]
    update_num_workers = toml_dict['strategy']["update_num_workers"]
    using_portfolio = toml_dict['strategy']["using_portfolio"]

    np.random.seed(seed)
    dim_change_func_list = [
        "MyAckley",
        "MyCosineMixture",
        "MyExponential",
        "MyGriewank",
        "MyLevyMontalvo1",
        "MyLevyMontalvo2",
        "MyNeumaier3",
        "MyPaviani",
        "MyRastrigin",
        "MyRosenbrock",
        "MySalomon",
        "MySchwefel",
        "MySinusoidal",
        "MyPutativeMichalewicz",
        "MyPutativeSineEnvelope",
        "MyPutativeEggHolder",
        "MyPutativeRana",
    ]
    assert func_name in dim_change_func_list
    assert in_dim is not None
    obj_obj = getattr(my_funcs, func_name)
    obj_func = obj_obj(dimensions=in_dim)
    rand_init = np.random.random(in_dim)
    # rand_init = np.ones(in_dim) * 0.5
    bounds_lower = obj_func.xmin
    bounds_upper = obj_func.xmax
    x_real = transform(rand_init, bounds_lower, bounds_upper)
    ref_start = obj_func(x_real)

    budget = (in_dim + 1) * budget_factor
    timebound = budget * func_sleep * 3
    assert timebound > 0
    f_star = obj_func.fglob 
    inits = [dict(candidate_value=retransform(init, bounds_lower, bounds_upper)) for init in inits_real]


    databases = []
    #results = []
    algo_num = len(algo_list)
    tot_num_workers = sum(num_worker_list)
    if algo_num == 1: # together makes no sense
        algo_name = algo_list[0]
        num_workers = num_worker_list[0]

        database = globals()["fun_"+algo_name](obj_func, rand_init, in_dim, bounds_lower, bounds_upper, func_sleep, budget, seed, num_workers, inits, timebound=timebound)
        databases.append(database)
        best_one = min([record['value'] for record in database  
                        if isinstance(record['value'], float)])
    elif not together:
        algo_name = "Or".join(algo_list) + "_".join([str(i) for i in num_worker_list])
        seed_list = [seed+i for i in range(algo_num)]
        unique_algo_list = ["{}_{}".format(index, algo) for index, algo in enumerate(algo_list)]
        opt_list = zip(algo_list, seed_list, num_worker_list, unique_algo_list)

        PROCESSES = algo_num
        print('Creating pool with %d processes\n' % PROCESSES)
        # with mp.Pool(PROCESSES) as pool:
        pool = NoDaemonProcessPool(PROCESSES)
        TASKS = [
                    (
                        globals()["fun_"+opt[0]], 
                        (obj_func, rand_init, in_dim, bounds_lower, bounds_upper, func_sleep, budget, opt[1], opt[2], inits, None, timebound)
                    ) 
                    for opt in opt_list
                ]
        print('Ordered results using pool.map() --- will block till complete:')
        for x in pool.map(process_runstar, TASKS):
            databases.append(x)
        best_one = min([
                            min([record['value'] for record in database  
                                 if isinstance(record['value'], float)])
                            for database in databases
                       ]
                   )
    else:
        algo_name = "Plus".join(algo_list) + "_".join([str(i) for i in num_worker_list])
        seed_list = [seed+i for i in range(algo_num)]
        pipe_list = [mp.Pipe() for _ in range(algo_num)]
        pconn_list = [pipe[0] for pipe in pipe_list]
        cconn_list = [pipe[1] for pipe in pipe_list]
        unique_algo_list = ["{}_{}".format(index, algo) for index, algo in enumerate(algo_list)]
        opt_list = zip(algo_list, seed_list, num_worker_list, cconn_list, unique_algo_list)

        PROCESSES = algo_num + 1
        print('Creating pool with %d processes\n' % PROCESSES)
        # with mp.Pool(PROCESSES) as pool:
        pool = NoDaemonProcessPool(PROCESSES)
        TASKS = [
                    (
                        globals()["fun_"+opt[0]], 
                        (obj_func, rand_init, in_dim, bounds_lower, bounds_upper, func_sleep, budget, opt[1], opt[2], inits, opt[3], None)
                    ) 
                    for opt in opt_list
                ] + \
                [
                    (
                        fun_feedback, 
                        (unique_algo_list, num_worker_list, pconn_list, in_dim, budget, pass_percent, pass_divide, seed, rand_init, ref_start, func_sleep, update_num_workers, using_portfolio)
                    )
                ]
        print('Ordered results using pool.map() --- will block till complete:')
        for x in pool.map(process_runstar, TASKS):
            databases.append(x)
        best_one = databases[-1]
    end_time = time.time()
    elapsed_time = end_time - start_time

    for index, (algo, num_workers) in enumerate(zip(algo_list, num_worker_list)):
        with open("database_{}_{}_{}.pkl".format(index, algo, num_workers), "wb") as f:
            pickle.dump(databases[index], f)

    loss =  best_one
    result = {"func_name": func_name + str(in_dim), 
              "dimension": in_dim, 
              "f_star": f_star, 
              "loss": best_one, 
              #"elapsed_budget": budget, 
              "budget_factor": budget_factor, 
              "elapsed_budget": budget*tot_num_workers, 
              "elapsed_time": elapsed_time, 
              "error": "",
              #"pseudotime": "",
              "num_objectives": 1}
    optim_setting = {#"budget": budget, 
                     #"num_workers": 1, 
                     "budget": budget*tot_num_workers,
                     "num_workers": tot_num_workers, 
                     "batch_mode": True, 
                     "optimizer_name": algo_name}
    summary = dict(result, seed=seed)
    #summary.update(obj_func.descriptors)
    summary.update(optim_setting)
    with open("summary.pkl", "wb") as f:
        pickle.dump(summary, f)
    open("finish", "a").close()
    print("run_goc.py done")
