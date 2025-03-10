from .model import ExactGPModel, SparseGPModel, train_GP, opt_acquisition
def suggest(train_x, train_y, inducing_x=None, device='cpu'):
    # Find optimal model hyperparameters
    train_x = train_x.to(device) 
    train_y = train_y.to(device) 
    if inducing_x is not None:
        inducing_x = inducing_x.to(device)
        model = SparseGPModel(
            train_x,
            train_y.squeeze(), 
            inducing_x
            ).to(device)
    else:
        model = ExactGPModel(
            train_x,
            train_y.squeeze(), 
            # inducing_x
            ).to(device)
    model.train()
    train_GP(
        model, 
        train_x, 
        train_y.squeeze(), 
        ) 
    # Optimize acquisition function
    # Get into evaluation (predictive posterior) mode
    model.eval().requires_grad_(False)
    new_x, min_acq = opt_acquisition(
        model, 
        train_x.shape[1],
        )
    return new_x.detach().cpu()