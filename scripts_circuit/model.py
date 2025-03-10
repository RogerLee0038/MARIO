import torch
from torch.distributions import Normal
# from tqdm import tqdm
import gpytorch
from utils import LRScheduler, EarlyStopping, ExpKernel

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
        self.covar_module = gpytorch.kernels.ScaleKernel(ExpKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernel())
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
    
    def train_hypers(self, train_iter=100):
        # self.train()
        # The "loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        # Training settings
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        lr_scheduler = LRScheduler(optimizer, patience=10, min_lr=1e-5, factor=0.1)
        early_stopping = EarlyStopping(patience=20)
        training_iter = train_iter
        emptyFlag = next(self.parameters()).is_cuda
        # Start iteration
        # iterator = tqdm(range(training_iter), desc='Train GP')
        # for _ in iterator:
        print("In train_hypers, enter training loop")
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
            if emptyFlag:
                torch.cuda.empty_cache()
            # iterator.set_postfix(loss=loss.item())
            lr_scheduler(loss.item())
            early_stopping(loss.item())
            if early_stopping.early_stop:
                print("In train_hypers, early stopped in iteration {}/{}".format(i+1, training_iter))
                break

    def calc_PI(self, input_x, best_one):
        best_y = self.y_transform(best_one)
        preds = self.posterior(input_x)
        mean = preds.mean
        var = preds.variance
        sigma = torch.sqrt(var) #+ 1e6
        PI = self.normal.cdf((best_y-mean) / sigma)
        return PI

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, num_tasks, normal=True, exp=True):
        if normal:
            self.train_x = train_x
            self.max_y = train_y.max()
            self.min_y = train_y.min()
            if self.max_y != self.min_y:
                self.train_y = (train_y-self.min_y)/(self.max_y-self.min_y)
            else:
                self.train_y = train_y
        else:
            self.train_x = train_x
            self.max_y = 0
            self.min_y = 0
            self.train_y = train_y
        super(MultitaskGPModel, self).__init__(self.train_x, self.train_y, 
        likelihood= gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks))
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        if exp:
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                ExpKernel(), num_tasks=num_tasks, rank=1
            )
        else:
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def posterior(self, *args, **kwargs):
        f_dist = super(MultitaskGPModel, self).__call__(*args, **kwargs)
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

    def train_hypers(self, train_iter=100):
        # self.train()
        # The "loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        # Training settings
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        lr_scheduler = LRScheduler(optimizer, patience=10, min_lr=1e-5, factor=0.1)
        early_stopping = EarlyStopping(patience=20)
        training_iter = train_iter
        emptyFlag = next(self.parameters()).is_cuda
        # Start iteration
        # iterator = tqdm(range(training_iter), desc='Train GP')
        # for _ in iterator:
        # print("In train_hypers, enter training loop")
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
            if emptyFlag:
                torch.cuda.empty_cache()
            # iterator.set_postfix(loss=loss.item())
            lr_scheduler(loss.item())
            early_stopping(loss.item())
            if early_stopping.early_stop:
                # print("In train_hypers, early stopped in iteration {}/{}".format(i+1, training_iter))
                break
