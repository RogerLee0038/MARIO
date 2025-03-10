from scipy.optimize import minimize
import torch
from torch.distributions import Normal
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from tqdm import tqdm
from .utils import LRScheduler, EarlyStopping

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        self.train_x = train_x
        self.max_y = train_y.max()
        self.min_y = train_y.min()
        if self.max_y != self.min_y:
            self.train_y = (train_y-self.min_y)/(self.max_y-self.min_y)
        else:
            self.train_y = train_y
        super(ExactGPModel, self).__init__(self.train_x, self.train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.best_y = self.train_y.min()
        self.normal = Normal(0,1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def posterior(self, *args, **kwargs):
        f_dist = super(ExactGPModel, self).__call__(*args, **kwargs)
        y_dist = self.likelihood(f_dist)
        return y_dist

    def y_transform(self, t):
        if self.max_y != self.min_y:
            return (t-self.min_y)/(self.max_y-self.min_y)
        else:
            return t

    def y_retransform(self, t):
        if self.max_y != self.min_y:
            return t * (self.max_y-self.min_y) + self.min_y
        else:
            return t

    def sigma_retransform(self, sigma):
        if self.max_y != self.min_y:
            return sigma * (self.max_y-self.min_y)
        else:
            return sigma

    def calc_EI(self, input_x):
        preds = self.posterior(input_x)
        mean = preds.mean
        var = preds.variance
        sigma = torch.sqrt(var) #+ 0.000001
        normed = -(mean - self.best_y) / sigma
        EI = sigma * (normed * self.normal.cdf(normed) + torch.exp(self.normal.log_prob(normed)))
        return EI

    def calc_PI(self, input_x):
        preds = self.posterior(input_x)
        mean = preds.mean
        var = preds.variance
        sigma = torch.sqrt(var) #+ 0.000001
        PI = self.normal.cdf(-mean/ sigma)
        return PI

    def calc_negUCB(self, input_x):
        preds = self.posterior(input_x)
        mean = preds.mean
        var = preds.variance
        sigma = torch.sqrt(var) #+ 0.000001
        UCB = -mean + 2.576*sigma
        return UCB

class SparseGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, inducing_x, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super(SparseGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=inducing_x.clone(), likelihood=likelihood)
        self.best_y = train_y.min()
        self.normal = Normal(0,1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(self, *args, **kwargs):
        f_dist = super(SparseGPModel, self).__call__(*args, **kwargs)
        y_dist = self.likelihood(f_dist)
        return y_dist

    def calc_EI(self, input_x):
        preds = self.posterior(input_x)
        mean = preds.mean
        var = preds.variance
        sigma = torch.sqrt(var) + 0.000001
        normed = -(mean - self.best_y) / sigma
        EI = sigma * (normed * self.normal.cdf(normed) + torch.exp(self.normal.log_prob(normed)))
        return EI

    def calc_PI(self, input_x):
        preds = self.posterior(input_x)
        mean = preds.mean
        var = preds.variance
        sigma = torch.sqrt(var) #+ 0.000001
        PI = self.normal.cdf(-mean/ sigma)
        return PI

    def calc_negUCB(self, input_x):
        preds = self.posterior(input_x)
        mean = preds.mean
        var = preds.variance
        sigma = torch.sqrt(var) #+ 0.000001
        UCB = -mean + 2.576*sigma
        return UCB

def train_GP(model, train_x, train_y): 
    ## Training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    ### if dim less than 100, training_iter=1000
    training_iter = max(10*train_x.shape[1], 1000)
    lr_scheduler = LRScheduler(optimizer, patience=5, min_lr=1e-4, factor=0.1)
    early_stopping = EarlyStopping(patience=10)
    ### "loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    ## Start iteration
    cuda_flag = False
    if next(model.parameters()).is_cuda:
        cuda_flag = True
    # model.train()
    # iterator = tqdm(range(training_iter), desc='Train GP')
    # for i in iterator:
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if cuda_flag:
            torch.cuda.empty_cache()
        # iterator.set_postfix(loss=loss.item())
        lr_scheduler(loss.item())
        early_stopping(loss.item())
        if early_stopping.early_stop:
            # print("in train_GP, early stopped in iteration {}/{}".format(i+1, training_iter))
            break

def opt_acquisition(model, in_dim):
    ## Optimizing settings
    ### if dim less than 100, acq_start_num=10000
    acq_start_num = max(100*in_dim, 10000)
    ### if dim less than 100, acq_tune_num=10
    acq_tune_num = max(int(0.1*in_dim), 10)
    acquisition = model.calc_negUCB

    ## Start warm up
    # model.eval().requires_grad_(False)
    model_device = next(model.parameters()).device
    acq_rand_inits = torch.rand(size=(acq_start_num,in_dim)).to(model_device)
    acq_ys = -acquisition(acq_rand_inits)
    candidate_x = []
    candidate_y = []
    for index in range(acq_tune_num):
        acq_arg = torch.argmin(acq_ys)
        acq_x = acq_rand_inits[acq_arg:acq_arg+1, : ]
        acq_y = acq_ys[acq_arg].clone()
        candidate_x.append(acq_x)
        candidate_y.append(acq_y)
        acq_ys[acq_arg] = torch.inf
    new_x = candidate_x[0]
    min_acq = candidate_y[0]

    ## Start fine tune
    def func(x):
        acq_x = torch.from_numpy(x).unsqueeze(dim=0).to(model_device).type(torch.float32)
        acq_y = -acquisition(acq_x)
        return acq_y.item()
    bounds = tuple([(0,1) for _ in range(in_dim)]) 
    for index in range(acq_tune_num):
        acq_init_x = candidate_x[index]
        acq_init_y = candidate_y[index]
        # print("optimize aquisition {}/{}, y starts with {}".format(index+1, acq_tune_num, acq_init_y.item()))
        init_x = acq_init_x.squeeze(dim=0).detach().cpu().numpy()
        res = minimize( func, 
                        init_x, 
                        bounds=bounds, 
                        method="L-BFGS-B")
        acq_x = torch.from_numpy(res.x).unsqueeze(axis=0).to(model_device).type(torch.float32)
        # print("optimize aquisition {}/{}, y ends with {}".format(index+1, acq_tune_num, res.fun))
        if res.fun < min_acq:
            new_x = acq_x
            min_acq = res.fun
        
    return new_x, min_acq
