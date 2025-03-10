import functools
import math
import warnings
import weakref
import numpy as np
from scipy import optimize as scipyoptimize
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.common import errors
from nevergrad.optimization import base
from nevergrad.optimization.base import IntOrParameter
from nevergrad.optimization import recaster
import sys
import os
sys.path.append(os.getenv('PO_ALGOS'))

class _MyNonObjectMinimizeBase(recaster.SequentialRecastOptimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        method: str = "Nelder-Mead",
        random_restart: bool = False,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.multirun = 1  # work in progress
        self._normalizer: tp.Any = None
        self.initial_guess: tp.Optional[tp.ArrayLike] = None #[0, 1]
        self.initials: tp.Optional[tp.ArrayLike] = None #[0,1]
        # configuration
        # assert method in ["Nelder-Mead", "COBYLA", "SLSQP", "Powell", "BOBYQA", "AX"], f"Unknown method '{method}'"
        # assert method in ["SMAC3", "SMAC", "Nelder-Mead", "COBYLA", "SLSQP", "Powell"], f"Unknown method '{method}'"
        self.method = method
        self.random_restart = random_restart
        # self._normalizer = p.helpers.Normalizer(self.parametrization)
        assert (
            method
            in [
                "CmaFmin2",
                "gomea",
                "gomeablock",
                "gomeatree",
                "SMAC3",
                "BFGS",
                "LBFGSB",
                "L-BFGS-B",
                "SMAC",
                "AX",
                "Lamcts",
                "Nelder-Mead",
                "COBYLA",
                "BOBYQA",
                "PYVTS",
                "PYNELDERMEAD",
                "SLSQP",
                "pysot",
                "negpysot",
                "Powell",
            ]
            or "NLOPT" in method
            or "BFGS" in method
        ), f"Unknown method '{method}'"
        if (
            method == "CmaFmin2"
            or "NLOPT" in method
            or "AX" in method
            or "BOBYQA" in method
            or "PYVTS" in method
            or "pysot" in method
            or "SMAC" in method
            or "L-BFGS-B" in method
            or method == "PYNELDERMEAD"
        ):
            normalizer = p.helpers.Normalizer(self.parametrization)
            #            if normalizer.fully_bounded or method == "AX" or "pysot" == method or "SMAC" in method:
            #                self._normalizer = normalizer
            self._normalizer = normalizer

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.Loss) -> None:
        """Called whenever calling "tell" on a candidate that was not "asked".
        Defaults to the standard tell pipeline.
        """  # We do not do anything; this just updates the current best.

    def get_optimization_function(self) -> tp.Callable[[tp.Callable[[tp.ArrayLike], float]], tp.ArrayLike]:
        return functools.partial(self._optimization_function, weakref.proxy(self))

    @staticmethod
    def _optimization_function(
        weakself: tp.Any, objective_function: tp.Callable[[tp.ArrayLike], float]
    ) -> tp.ArrayLike:
        # pylint:disable=unused-argument
        budget = np.inf if weakself.budget is None else weakself.budget
        best_res = np.inf
        best_x: np.ndarray = weakself.current_bests["average"].x
        if weakself.initial_guess is not None:
            best_x = np.array(weakself.initial_guess, copy=True)  # copy, just to make sure it is not modified

        remaining: float = budget - weakself._num_ask

        def ax_obj(p):
            data = [p["x" + str(i)] for i in range(weakself.dimension)]  # type: ignore
            if weakself._normalizer:
                data = weakself._normalizer.backward(np.asarray(data, dtype=np.float_))
            return objective_function(data)

        while remaining > 0:  # try to restart if budget is not elapsed
            # print(f"Iteration with remaining={remaining}")
            options: tp.Dict[str, tp.Any] = {} if weakself.budget is None else {"maxiter": remaining}
            if weakself.method == "BOBYQA":
                import mypybobyqa  # type: ignore

                def pybobyqa_func(x):
                    assert len(x) == weakself.dimension, (
                        str(x) + " does not have length " + str(weakself.dimension)
                    )
                    # x in [0,1], back to the std data, 
                    # which is set into the nevergrad bounds again in recaster.py
                    if weakself._normalizer is not None:
                        data = weakself._normalizer.backward(np.asarray(x, dtype=np.float32))
                    # Fake function, send data to the ask queue and wait for the tell
                    return objective_function(data)

                if weakself.initials is not None:
                    x_data = np.array([tup[0] for tup in weakself.initials])
                    rhobeg_appoxi = np.max(np.max(np.abs(x_data - best_x), axis=0))
                    rhobeg_appoxi = min(max(rhobeg_appoxi, 0.01), 0.4)
                else:
                    rhobeg_appoxi = 0.1
                res = mypybobyqa.solve(
                    pybobyqa_func, 
                    best_x, # get from initial_guess, [0,1]
                    initials = weakself.initials, 
                    bounds = ([0]*weakself.dimension, [1]*weakself.dimension),
                    npt = 2 * weakself.dimension + 1,
                    rhobeg = rhobeg_appoxi,
                    maxfun = budget, 
                    seek_global_minimum = True,
                    do_logging = True
                )

                if res.f < best_res:
                    best_res = res.f
                    best_x = res.x

            elif weakself.method == "PYVTS":
                from VTS import myVTS # type: ignore

                def pyvts_func(x):
                    assert len(x) == weakself.dimension, (
                        str(x) + " does not have length " + str(weakself.dimension)
                    )
                    # x in [0,1], back to the std data, 
                    # which is set into the nevergrad bounds again in recaster.py
                    if weakself._normalizer is not None:
                        data = weakself._normalizer.backward(np.asarray(x, dtype=np.float32))
                    # Fake function, send data to the ask queue and wait for the tell
                    return objective_function(data)

                pyvts_opt = myVTS(
                             np.zeros(weakself.dimension),              
                             # the lower bound of each problem dimensions
                             np.ones(weakself.dimension),              
                             # the upper bound of each problem dimensions
                             weakself.dimension,          
                             # the problem dimensions
                             ninits = 2*weakself.dimension+1,      
                             # the number of random samples used in initializations 
                             init = best_x,
                             with_init = True,
                             initials = weakself.initials,
                             func = pyvts_func,               
                             # function object to be optimized
                             iteration = budget,    
                             # maximal iteration to evaluate f(x)
                             Cp = 0,              
                             # Cp for UCT function
                             leaf_size = 4*weakself.dimension,  # tree leaf size
                             kernel_type = 'rbf',   # kernel for GP model
                             use_cuda = False,     #train GP on GPU
                             set_greedy = True    # greedy setting
                             )
                res_f, res_x = pyvts_opt.search()

                if res_f < best_res:
                    best_res = res_f
                    best_x = res_x

            elif weakself.method == "PYNELDERMEAD":
                import myneldermead# type: ignore

                def pyneldermead_func(x):
                    x = np.clip(x, 0, 1)
                    assert len(x) == weakself.dimension, (
                        str(x) + " does not have length " + str(weakself.dimension)
                    )
                    # x in [0,1], back to the std data, 
                    # which is set into the nevergrad bounds again in recaster.py
                    print(x)
                    if weakself._normalizer is not None:
                        data = weakself._normalizer.backward(np.asarray(x, dtype=np.float32))
                    print(data)
                    # Fake function, send data to the ask queue and wait for the tell
                    return objective_function(data)

                if weakself.initials is not None:
                    x_data = np.array([tup[0] for tup in weakself.initials])
                    rhobeg_appoxi = np.max(np.max(np.abs(x_data - best_x), axis=0))
                    rhobeg_appoxi = min(max(rhobeg_appoxi, 0.01), 0.4)
                else:
                    rhobeg_appoxi = 0.1
                res_x, res_f = myneldermead.solve(
                    pyneldermead_func, 
                    best_x, # get from initial_guess, [0,1]
                    step = rhobeg_appoxi, 
                    max_eval = budget,
                    initials = weakself.initials, 
                )

                if res_f < best_res:
                    best_res = res_f
                    best_x = res_x

            elif weakself.method == "AX":
                from ax import optimize as axoptimize  # type: ignore

                parameters = [
                    {"name": "x" + str(i), "type": "range", "bounds": [0.0, 1.0]}
                    for i in range(weakself.dimension)
                ]
                best_parameters, _best_values, _experiment, _model = axoptimize(
                    parameters, evaluation_function=ax_obj, minimize=True, total_trials=budget
                )
                best_x = np.array([float(best_parameters["x" + str(i)]) for i in range(weakself.dimension)])
                best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=float))
            # options: tp.Dict[str, tp.Any] = {} if weakself.budget is None else {"maxiter": remaining}
            elif weakself.method[:5] == "NLOPT":
                # This is NLOPT, used as in the PCSE simulator notebook.
                # ( https://github.com/ajwdewit/pcse_notebooks ).
                import nlopt  # type: ignore

                def nlopt_objective_function(*args):
                    try:
                        data = np.asarray([arg for arg in args if len(arg) > 0])[0]
                    except Exception as e:
                        raise ValueError(f"{e}:\n{args}\n {[arg for arg in args]}")
                    assert len(data) == weakself.dimension, (
                        str(data) + " does not have length " + str(weakself.dimension)
                    )
                    if weakself._normalizer is not None:
                        data = weakself._normalizer.backward(np.asarray(data, dtype=np.float32))
                    return objective_function(data)

                # Sbplx (based on Subplex) is used by default.
                nlopt_param = (
                    getattr(nlopt, weakself.method[6:]) if len(weakself.method) > 5 else nlopt.LN_SBPLX
                )
                opt = nlopt.opt(nlopt_param, weakself.dimension)
                # Assign the objective function calculator
                opt.set_min_objective(nlopt_objective_function)
                # Set the bounds.
                opt.set_lower_bounds(np.zeros(weakself.dimension))
                opt.set_upper_bounds(np.ones(weakself.dimension))
                # opt.set_initial_step([0.05, 0.05])
                opt.set_maxeval(budget)

                # Start the optimization with the first guess
                # firstguess = 0.5 * np.ones(weakself.dimension)
                firstguess = best_x
                best_x = opt.optimize(firstguess)
                # print("\noptimum at TDWI: %s, SPAN: %s" % (x[0], x[1]))
                # print("minimum value = ",  opt.last_optimum_value())
                # print("result code = ", opt.last_optimize_result())
                # print("With %i function calls" % objfunc_calculator.n_calls)
                ## if weakself._normalizer is not None:
                ##     best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=np.float32))
            elif "pysot" in weakself.method:
                from poap.controller import BasicWorkerThread, ThreadController  # type: ignore

                from pySOT.experimental_design import SymmetricLatinHypercube  # type: ignore
                from pySOT.optimization_problems import OptimizationProblem  # type: ignore

                # from pySOT.strategy import SRBFStrategy
                from pySOT.strategy import DYCORSStrategy  # type: ignore
                from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant  # type: ignore

                class LocalOptimizationProblem(OptimizationProblem):
                    def eval(self, data):
                        if weakself._normalizer is not None:
                            data = weakself._normalizer.backward(np.asarray(data, dtype=np.float32))
                        val = (
                            float(objective_function(data))
                            if "negpysot" not in weakself.method
                            else -float(objective_function(data))
                        )
                        return val

                dim = weakself.dimension
                opt_prob = LocalOptimizationProblem()
                opt_prob.dim = dim
                opt_prob.lb = np.array([0.0] * dim)
                opt_prob.ub = np.array([1.0] * dim)
                opt_prob.int_var = []
                opt_prob.cont_var = np.array(range(dim))

                rbf = RBFInterpolant(
                    dim=opt_prob.dim,
                    lb=opt_prob.lb,
                    ub=opt_prob.ub,
                    kernel=CubicKernel(),
                    tail=LinearTail(opt_prob.dim),
                )
                slhd = SymmetricLatinHypercube(dim=opt_prob.dim, num_pts=2 * (opt_prob.dim + 1))
                controller = ThreadController()
                # controller.strategy = SRBFStrategy(
                #    max_evals=budget, opt_prob=opt_prob, exp_design=slhd, surrogate=rbf, asynchronous=True
                # )
                controller.strategy = DYCORSStrategy(
                    opt_prob=opt_prob, exp_design=slhd, surrogate=rbf, max_evals=budget, asynchronous=True
                )
                worker = BasicWorkerThread(controller, opt_prob.eval)
                controller.launch_worker(worker)

                result = controller.run()

                best_res = result.value
                best_x = result.params[0]

            elif weakself.method == "SMAC3":

                # Import ConfigSpace and different types of parameters
                # from smac.configspace import ConfigurationSpace  # type: ignore  # noqa  # pylint: disable=unused-import
                # from smac.configspace import UniformFloatHyperparameter  # type: ignore
                # from smac.facade.smac_hpo_facade import SMAC4HPO  # type: ignore  # noqa  # pylint: disable=unused-import

                from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
                from smac import HyperparameterOptimizationFacade, Scenario

                # Import SMAC-utilities
                import threading
                import os
                import time
                from pathlib import Path

                the_date = str(time.time()) + "_" + str(np.random.rand())
                tag = str(np.random.rand())
                feed = "/tmp/smac_feed" + the_date + ".txt"
                fed = "/tmp/smac_fed" + the_date + ".txt"

                def dummy_function():
                    for _ in range(remaining):
                        # print(f"side thread waiting for request... ({u}/{weakself.budget})")
                        while (not Path(feed).is_file()) or os.stat(feed).st_size == 0:
                            time.sleep(0.1)
                        time.sleep(0.1)
                        # print("side thread happy to work on a request...")
                        data = np.loadtxt(feed)
                        os.remove(feed)
                        # print("side thread happy to really work on a request...")
                        res = objective_function(data)
                        # print("side thread happy to forward the result of a request...")
                        f = open(fed, "w")
                        f.write(str(res))
                        f.close()
                    return

                thread = threading.Thread(target=dummy_function)
                thread.start()

                # print(f"start SMAC3 optimization with budget {budget} in dimension {weakself.dimension}")
                cs = ConfigurationSpace()
                cs.add_hyperparameters(
                    [
                        UniformFloatHyperparameter(f"x{tag}{i}", 0.0, 1.0, default_value=0.0)
                        for i in range(weakself.dimension)
                    ]
                )

                def smac2_obj(p, seed: int = 0):
                    # print(f"SMAC3 proposes {p} {type(p)}")
                    pdata = [p[f"x{tag}{i}"] for i in range(len(p.keys()))]
                    data = weakself._normalizer.backward(np.asarray(pdata, dtype=float))
                    # print(f"converted to {data}")
                    if Path(fed).is_file():
                        os.remove(fed)
                    np.savetxt(feed, data)
                    while (not Path(fed).is_file()) or os.stat(fed).st_size == 0:
                        time.sleep(0.1)
                    time.sleep(0.1)
                    f = open(fed, "r")
                    res = float(f.read())
                    f.close()
                    # print(f"SMAC3 will receive {res}")
                    return res

                # scenario = Scenario({'cs': cs, 'run_obj': smac2_obj, 'runcount-limit': remaining, 'deterministic': True})
                scenario = Scenario(cs, deterministic=True, n_trials=int(remaining))

                smac = HyperparameterOptimizationFacade(scenario, smac2_obj)
                res = smac.optimize()
                best_x = np.array([res[f"x{tag}{k}"] for k in range(len(res.keys()))])
                best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=float))
                # print(f"end SMAC optimization {best_x}")
                thread.join()
                weakself._num_ask = budget

            #            elif weakself.method == "SMAC":
            #                import smac  # noqa  # pylint: disable=unused-import
            #                import scipy.optimize  # noqa  # pylint: disable=unused-import
            #                from smac.facade.func_facade import fmin_smac  # noqa  # pylint: disable=unused-import
            #
            #                import threading
            #                import os
            #                import time
            #                from pathlib import Path
            #
            #                the_date = str(time.time())
            #                feed = "/tmp/smac_feed" + the_date + ".txt"
            #                fed = "/tmp/smac_fed" + the_date + ".txt"
            #
            #                def dummy_function():
            #                    for u in range(remaining):
            #                        print(f"side thread waiting for request... ({u}/{weakself.budget})")
            #                        while (not Path(feed).is_file()) or os.stat(feed).st_size == 0:
            #                            time.sleep(0.1)
            #                        time.sleep(0.1)
            #                        print("side thread happy to work on a request...")
            #                        data = np.loadtxt(feed)
            #                        os.remove(feed)
            #                        print("side thread happy to really work on a request...")
            #                        res = objective_function(data)
            #                        print("side thread happy to forward the result of a request...")
            #                        f = open(fed, "w")
            #                        f.write(str(res))
            #                        f.close()
            #                    return
            #
            #                thread = threading.Thread(target=dummy_function)
            #                thread.start()
            #
            #                def smac_obj(p):
            #                    print(f"SMAC proposes {p}")
            #                    data = weakself._normalizer.backward(
            #                        np.asarray([p[i] for i in range(len(p))], dtype=np.float)
            #                    )
            #                    print(f"converted to {data}")
            #                    if Path(fed).is_file():
            #                        os.remove(fed)
            #                    np.savetxt(feed, data)
            #                    while (not Path(fed).is_file()) or os.stat(fed).st_size == 0:
            #                        time.sleep(0.1)
            #                    time.sleep(0.1)
            #                    f = open(fed, "r")
            #                    res = np.float(f.read())
            #                    f.close()
            #                    print(f"SMAC will receive {res}")
            #                    return res
            #
            #                print(f"start SMAC optimization with budget {budget} in dimension {weakself.dimension}")
            #                assert budget is not None
            #                x, cost, _ = fmin_smac(
            #                    # func=lambda x: sum([(x_ - 1.234)**2  for x_ in x]),
            #                    func=smac_obj,
            #                    x0=[0.0] * weakself.dimension,
            #                    bounds=[(0.0, 1.0)] * weakself.dimension,
            #                    maxfun=remaining,
            #                    rng=weakself._rng.randint(5000),
            #                )  # Passing a seed makes fmin_smac determistic
            #                print("end SMAC optimization")
            #                thread.join()
            #                weakself._num_ask = budget
            #
            #                if cost < best_res:
            #                    best_res = cost
            #                    best_x = weakself._normalizer.backward(np.asarray(x, dtype=float))
            #

            elif "gomea" in weakself.method:
                import gomea

                class gomea_function(gomea.fitness.BBOFitnessFunctionRealValued):
                    def objective_function(self, objective_index, data):  # type: ignore
                        if weakself._normalizer is not None:
                            data = weakself._normalizer.backward(np.asarray(data, dtype=np.float32))
                        return objective_function(data)

                gomea_f = gomea_function(weakself.dimension)
                lm = {
                    "gomea": gomea.linkage.Univariate(),
                    "gomeablock": gomea.linkage.BlockMarginalProduct(2),
                    "gomeatree": gomea.linkage.LinkageTree("NMI".encode(), True, 0),
                }[weakself.method]
                rvgom = gomea.RealValuedGOMEA(
                    fitness=gomea_f,
                    linkage_model=lm,
                    lower_init_range=0.0,
                    upper_init_range=1.0,
                    max_number_of_evaluations=budget,
                )
                rvgom.run()
                best_x = gomea_f.best_x

            elif weakself.method == "CmaFmin2":
                import cma  # type: ignore

                def cma_objective_function(data):
                    # Hopefully the line below does nothing if unbounded and rescales from [0, 1] if bounded.
                    if weakself._normalizer is not None and weakself._normalizer.fully_bounded:
                        data = weakself._normalizer.backward(np.asarray(data, dtype=np.float32))
                    return objective_function(data)

                # cma.fmin2(objective_function, [0.0] * self.dimension, [1.0] * self.dimension, remaining)
                x0 = (
                    0.5 * np.ones(weakself.dimension)
                    if weakself._normalizer is not None and weakself._normalizer.fully_bounded
                    else np.zeros(weakself.dimension)
                )
                num_calls = 0
                while budget - num_calls > 0:
                    options = {"maxfevals": budget - num_calls, "verbose": -9}
                    if weakself._normalizer is not None and weakself._normalizer.fully_bounded:
                        # Tell CMA to work in [0, 1].
                        options["bounds"] = [0.0, 1.0]
                    res = cma.fmin(
                        cma_objective_function,
                        x0=x0,
                        sigma0=0.2,
                        options=options,
                        restarts=9,
                    )
                    x0 = (
                        0.5
                        + np.random.uniform() * np.random.uniform(low=-0.5, high=0.5, size=weakself.dimension)
                        if weakself._normalizer is not None and weakself._normalizer.fully_bounded
                        else np.random.randn(weakself.dimension)
                    )
                    if res[1] < best_res:
                        best_res = res[1]
                        best_x = res[0]
                        if weakself._normalizer is not None:
                            best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=np.float32))
                    num_calls += res[2]
            elif weakself.method == "L-BFGS-B":
                def lbfgsb_objective_function(data):
                    assert len(data) == weakself.dimension, (
                        str(data) + " does not have length " + str(weakself.dimension)
                    )
                    if weakself._normalizer is not None:
                        data = weakself._normalizer.backward(np.asarray(data, dtype=np.float32))
                    return objective_function(data)
                res = scipyoptimize.minimize(
                    lbfgsb_objective_function,
                    best_x
                    if not weakself.random_restart
                    else weakself._rng.normal(0.0, 1.0, weakself.dimension),
                    method=weakself.method,
                    bounds = tuple([(0,1) for _ in range(weakself.dimension)]),
                    options=options,
                    tol=0,
                )
                if res.fun < best_res:
                    best_res = res.fun
                    best_x = res.x
            else:
                res = scipyoptimize.minimize(
                    objective_function,
                    best_x
                    if not weakself.random_restart
                    else weakself._rng.normal(0.0, 1.0, weakself.dimension),
                    method=weakself.method,
                    options=options,
                    tol=0,
                )
                if res.fun < best_res:
                    best_res = res.fun
                    best_x = res.x
            remaining = budget - weakself._num_ask
        assert best_x is not None
        return best_x


class MyNonObjectOptimizer(base.ConfiguredOptimizer):
    """Wrapper over Scipy optimizer implementations, in standard ask and tell format.
    This is actually an import from scipy-optimize, including Sequential Quadratic Programming,

    Parameters
    ----------
    method: str
        Name of the method to use among:

        - Nelder-Mead
        - COBYLA
        - SQP (or SLSQP): very powerful e.g. in continuous noisy optimization. It is based on
          approximating the objective function by quadratic models.
        - Powell
        - NLOPT* (https://nlopt.readthedocs.io/en/latest/; by default, uses Sbplx, based on Subplex);
            can be NLOPT,
                NLOPT_LN_SBPLX,
                NLOPT_LN_PRAXIS,
                NLOPT_GN_DIRECT,
                NLOPT_GN_DIRECT_L,
                NLOPT_GN_CRS2_LM,
                NLOPT_GN_AGS,
                NLOPT_GN_ISRES,
                NLOPT_GN_ESCH,
                NLOPT_LN_COBYLA,
                NLOPT_LN_BOBYQA,
                NLOPT_LN_NEWUOA_BOUND,
                NLOPT_LN_NELDERMEAD.
    random_restart: bool
        whether to restart at a random point if the optimizer converged but the budget is not entirely
        spent yet (otherwise, restarts from best point)

    Note
    ----
    These optimizers do not support asking several candidates in a row
    """

    recast = True
    no_parallelization = True

    # pylint: disable=unused-argument
    def __init__(self, *, method: str = "Nelder-Mead", random_restart: bool = False) -> None:
        super().__init__(_MyNonObjectMinimizeBase, locals())

My_NLOPT_LN_BOBYQA = MyNonObjectOptimizer(method="NLOPT_LN_BOBYQA").set_name("My_NLOPT_LN_BOBYQA", register=True)
My_NLOPT_LN_NELDERMEAD= MyNonObjectOptimizer(method="NLOPT_LN_NELDERMEAD").set_name("My_NLOPT_LN_NELDERMEAD", register=True)
My_NLOPT_GN_DIRECT= MyNonObjectOptimizer(method="NLOPT_GN_DIRECT").set_name("My_NLOPT_GN_DIRECT", register=True)
My_LBFGSB = MyNonObjectOptimizer(method="L-BFGS-B", random_restart=False).set_name("My_LBFGSB", register=True)
My_PYBOBYQA = MyNonObjectOptimizer(method="BOBYQA").set_name("My_PYBOBYQA", register=True)
My_PYVTS = MyNonObjectOptimizer(method="PYVTS").set_name("My_PYVTS", register=True)
My_PYNELDERMEAD= MyNonObjectOptimizer(method="PYNELDERMEAD").set_name("My_PYNELDERMEAD", register=True)

# AX = NonObjectOptimizer(method="AX").set_name("AX", register=True)
# BOBYQA = NonObjectOptimizer(method="BOBYQA").set_name("BOBYQA", register=True)
# NelderMead = NonObjectOptimizer(method="Nelder-Mead").set_name("NelderMead", register=True)
# CmaFmin2 = NonObjectOptimizer(method="CmaFmin2").set_name("CmaFmin2", register=True)
# GOMEA = NonObjectOptimizer(method="gomea").set_name("GOMEA", register=True)
# GOMEABlock = NonObjectOptimizer(method="gomeablock").set_name("GOMEABlock", register=True)
# GOMEATree = NonObjectOptimizer(method="gomeatree").set_name("GOMEATree", register=True)
# # NLOPT = NonObjectOptimizer(method="NLOPT").set_name("NLOPT", register=True)
# Powell = NonObjectOptimizer(method="Powell").set_name("Powell", register=True)
# RPowell = NonObjectOptimizer(method="Powell", random_restart=True).set_name("RPowell", register=True)
# BFGS = NonObjectOptimizer(method="BFGS", random_restart=True).set_name("BFGS", register=True)
# LBFGSB = NonObjectOptimizer(method="L-BFGS-B", random_restart=True).set_name("LBFGSB", register=True)
# Cobyla = NonObjectOptimizer(method="COBYLA").set_name("Cobyla", register=True)
# RCobyla = NonObjectOptimizer(method="COBYLA", random_restart=True).set_name("RCobyla", register=True)
# SQP = NonObjectOptimizer(method="SLSQP").set_name("SQP", register=True)
# SLSQP = SQP  # Just so that people who are familiar with SLSQP naming are not lost.
# RSQP = NonObjectOptimizer(method="SLSQP", random_restart=True).set_name("RSQP", register=True)
# RSLSQP = RSQP  # Just so that people who are familiar with SLSQP naming are not lost.
# # NEWUOA = NonObjectOptimizer(method="NLOPT_LN_NEWUOA_BOUND").set_name("NEWUOA", register=True)
# NLOPT_LN_SBPLX = NonObjectOptimizer(method="NLOPT_LN_SBPLX").set_name("NLOPT_LN_SBPLX", register=True)
# NLOPT_LN_PRAXIS = NonObjectOptimizer(method="NLOPT_LN_PRAXIS").set_name("NLOPT_LN_PRAXIS", register=True)
# NLOPT_GN_DIRECT = NonObjectOptimizer(method="NLOPT_GN_DIRECT").set_name("NLOPT_GN_DIRECT", register=True)
# NLOPT_GN_DIRECT_L = NonObjectOptimizer(method="NLOPT_GN_DIRECT_L").set_name(
#     "NLOPT_GN_DIRECT_L", register=True
# )
# NLOPT_GN_CRS2_LM = NonObjectOptimizer(method="NLOPT_GN_CRS2_LM").set_name("NLOPT_GN_CRS2_LM", register=True)
# NLOPT_GN_AGS = NonObjectOptimizer(method="NLOPT_GN_AGS").set_name("NLOPT_GN_AGS", register=True)
# NLOPT_GN_ISRES = NonObjectOptimizer(method="NLOPT_GN_ISRES").set_name("NLOPT_GN_ISRES", register=True)
# NLOPT_GN_ESCH = NonObjectOptimizer(method="NLOPT_GN_ESCH").set_name("NLOPT_GN_ESCH", register=True)
# NLOPT_LN_COBYLA = NonObjectOptimizer(method="NLOPT_LN_COBYLA").set_name("NLOPT_LN_COBYLA", register=True)
# NLOPT_LN_BOBYQA = NonObjectOptimizer(method="NLOPT_LN_BOBYQA").set_name("NLOPT_LN_BOBYQA", register=True)
# NLOPT_LN_NEWUOA_BOUND = NonObjectOptimizer(method="NLOPT_LN_NEWUOA_BOUND").set_name(
#     "NLOPT_LN_NEWUOA_BOUND", register=True
# )
# NLOPT_LN_NELDERMEAD = NonObjectOptimizer(method="NLOPT_LN_NELDERMEAD").set_name(
#     "NLOPT_LN_NELDERMEAD", register=True
# )
# # AX = NonObjectOptimizer(method="AX").set_name("AX", register=True)
# # BOBYQA = NonObjectOptimizer(method="BOBYQA").set_name("BOBYQA", register=True)
# # SMAC = NonObjectOptimizer(method="SMAC").set_name("SMAC", register=True)
# SMAC3 = NonObjectOptimizer(method="SMAC3").set_name("SMAC3", register=True)
