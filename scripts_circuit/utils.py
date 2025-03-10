import numpy as np
import time
from concurrent import futures

import math
from typing import Optional

import torch
from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
from gpytorch.kernels.kernel import Kernel

def retransform(x_real:np.ndarray, bounds_lower, bounds_upper):
    mask = bounds_lower != bounds_upper # == is possible if update_bounds
    x_temp = x_real
    x_temp[mask] = (x_real[mask]-bounds_lower[mask])/(bounds_upper[mask]-bounds_lower[mask])
    x_temp[~mask] = 0
    x_01 = x_temp.clip(0,1)
    return x_01

def transform(x_01:np.ndarray, bounds_lower, bounds_upper):
    x_real = x_01 * (bounds_upper-bounds_lower) + bounds_lower
    return x_real

class Func: # avoid closure for multiprocessing
    def __init__(self, obj_func, func_sleep):
        self.obj_func = obj_func
        self.func_sleep = func_sleep
    def __call__(self, x_real: np.ndarray, index: int) -> float:
        y = self.obj_func(x_real, index=index)
        time.sleep(self.func_sleep)
        return y

def run_funcs(func, x_real_chunk, num_workers, totCnt):
    # assert len(x_real_chunk) == num_workers
    results_list = []
    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        running_jobs = []
        finished_jobs = []
        for index, x_real in enumerate(x_real_chunk):
            running_jobs.append(
                (index, x_real, executor.submit(func, x_real, totCnt+index))
            )
        while running_jobs or finished_jobs:
            if finished_jobs:
                for index, x_real, job in finished_jobs:
                    value = job.result()
                    results_list.append((index, x_real, value))
                finished_jobs = []
            tmp_runnings, tmp_finished = [], []
            for index, x_real, job in running_jobs:
                (tmp_finished if job.done() else tmp_runnings).append((index, x_real, job))
            running_jobs, finished_jobs = tmp_runnings, tmp_finished
    fx_chunk = [results[2] for results in sorted(results_list, key=lambda i:i[0])]
    totCnt += len(fx_chunk)
    return fx_chunk, totCnt

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            #print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                #print('INFO: Early stopping')
                self.early_stop = True

class ExpKernel(Kernel):
    """
    exp kernel from FTBO
    """
    is_stationary = True

    def __init__(
        self,
        exp_alpha_prior: Optional[Prior] = None,
        exp_alpha_constraint: Optional[Interval] = None,
        exp_beta_prior: Optional[Prior] = None,
        exp_beta_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(ExpKernel, self).__init__(**kwargs)

        self.register_parameter(
            name="raw_exp_alpha", parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name="raw_exp_beta", parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, 1))
        )

        if exp_alpha_constraint is None:
            exp_alpha_constraint = Positive()
        if exp_beta_constraint is None:
            exp_beta_constraint = Positive()

        if exp_alpha_prior is not None:
            if not isinstance(exp_alpha_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(exp_alpha_prior).__name__)
            self.register_prior(
                "exp_alpha_prior",
                exp_alpha_prior,
                lambda m: m.exp_alpha,
                lambda m, v: m._set_exp_alpha(v),
            )
        if exp_beta_prior is not None:
            if not isinstance(exp_beta_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(exp_beta_prior).__name__)
            self.register_prior(
                "exp_beta_prior",
                exp_beta_prior,
                lambda m: m.exp_beta,
                lambda m, v: m._set_exp_beta(v),
            )

        self.register_constraint("raw_exp_alpha", exp_alpha_constraint)
        self.register_constraint("raw_exp_beta", exp_beta_constraint)

    @property
    def exp_alpha(self):
        return self.raw_exp_alpha_constraint.transform(self.raw_exp_alpha)

    @exp_alpha.setter
    def exp_alpha(self, value):
        return self._set_exp_alpha(value)
    def _set_exp_alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_exp_alpha)
        self.initialize(raw_exp_alpha=self.raw_exp_alpha_constraint.inverse_transform(value))

    @property
    def exp_beta(self):
        return self.raw_exp_beta_constraint.transform(self.raw_exp_beta)

    @exp_beta.setter
    def exp_beta(self, value):
        return self._set_exp_beta(value)
    def _set_exp_beta(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_exp_beta)
        self.initialize(raw_exp_beta=self.raw_exp_beta_constraint.inverse_transform(value))

    def add_dist(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        square_dist: bool = False,
        **params,
    ) -> torch.Tensor:
        r"""
        This is a helper method for computing the addition between
        all pairs of points in :math:`\mathbf x_1` and :math:`\mathbf x_2`.

        :param x1: First set of data (... x N x D).
        :param x2: Second set of data (... x M x D).
        :param diag: Should the Kernel compute the whole kernel, or just the diag?
            If True, it must be the case that `x1 == x2`. (Default: False.)
        :param last_dim_is_batch: If True, treat the last dimension
            of `x1` and `x2` as another batch dimension.
            (Useful for additive structure over the dimensions). (Default: False.)
        :param square_dist:
            If True, returns the squared distance rather than the standard distance. (Default: False.)
        :return: The kernel matrix or vector. The shape depends on the kernel's evaluation mode:

            * `full_covar`: `... x N x M`
            * `full_covar` with `last_dim_is_batch=True`: `... x K x N x M`
            * `diag`: `... x N`
            * `diag` with `last_dim_is_batch=True`: `... x K x N`
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        res = None

        if diag:
            res = (x1 + x2).squeeze(-1)
            return res
        else:
            x2 = x2.transpose(-1,-2)
            res = x1 + x2
            return res

    def forward(self, x1, x2, diag=False, **params):
        assert x1.shape[-1] == 1
        assert x2.shape[-1] == 1
        covar_add = self.add_dist(x1, x2, diag=diag, **params)
        deno = torch.pow(covar_add+self.exp_beta, self.exp_alpha)
        mole = torch.pow(self.exp_beta, self.exp_alpha)
        diff = mole/deno
        if diag:
            return diff.squeeze(0)
        else:
            return diff