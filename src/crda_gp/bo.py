import botorch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch

from botorch.models.transforms import Normalize, Standardize
from botorch.models.gp_regression import SingleTaskGP
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval

"""Implementation of the Bayesian Optimization from the Gauche Bayesian Optimization tutorial
https://leojklarner.github.io/gauche/notebooks/bayesian_optimisation_over_molecules.html"""

class TanimotoGP(SingleTaskGP):
    def __init__(self, 
                 train_x: torch.Tensor, 
                 train_y: torch.Tensor, 
                 likelihood: gpytorch.likelihoods=GaussianLikelihood()):
        super().__init__(train_X=train_x, train_Y=train_y, likelihood=likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
        self.to(train_x)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def initialize_model(train_x:torch.Tensor, 
                     train_obj: torch.Tensor, 
                     state_dict=None):

    # define model for objective
    model = TanimotoGP(train_x=train_x, train_y=train_obj).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    
    return model, mll

def initialize_model_dense(train_x:torch.Tensor, 
                     train_obj: torch.Tensor, 
                     state_dict=None):

    train_obj = train_obj

    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=train_x.shape[-1]
        )
    )

    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_obj,
        covar_module=covar_module,
        input_transform=Normalize(
            d=train_x.shape[-1]
        )
    )

    if state_dict is not None:
        model.load_state_dict(state_dict)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    return mll, model

def optimize_acqf_and_get_observation(acq_func:botorch.acquisition, 
                                      heldout_inputs:torch.Tensor, 
                                      heldout_outputs:torch.Tensor):
    # Loop over the discrete set of points to evaluate the acquisition function at.
    acq_vals = []
    for i in range(len(heldout_outputs)):
        acq_vals.append(acq_func(heldout_inputs[i].unsqueeze(-2)))

    # Observe new values
    acq_vals = torch.stack(acq_vals)
    best_idx = torch.argmax(acq_vals)
    new_x = heldout_inputs[best_idx].unsqueeze(-2) # add batch dimension
    new_obj = heldout_outputs[best_idx].unsqueeze(-1) # add output dimension

    # Delete the selected input and value from the heldout
    heldout_inputs = torch.cat((heldout_inputs[:best_idx], heldout_inputs[best_idx+1:]), axis=0)
    heldout_outputs = torch.cat((heldout_outputs[:best_idx], heldout_outputs[best_idx+1:]), axis=0)

    return new_x, new_obj, heldout_inputs, heldout_outputs

def plot_confidence(best_observed: list,
                    N_ITERS:int=20):
    """Plots the Cofidence Intervals of the y plot per iteration"""
    iters = np.arange(N_ITERS + 1)
    y_act = np.asarray(torch.stack(best_observed))
    y_act_mean = y_act.mean(axis=0)
    y_act_std = y_act.std(axis=0)
    lower = y_act_mean - y_act_std
    upper = y_act_mean + y_act_std

    # print(f"Y Act: {y_act}")
    # print(f"Y Act Mean: {y_act_mean}")
    # print(f"Y Act Std: {y_act_std}")

    plt.plot(iters, y_act_mean, label='ACT')
    plt.fill_between(iters, lower, upper, alpha=0.2)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Best Objective Value')
    plt.legend(loc="lower right")
    plt.xticks(list(np.arange(1, N_ITERS+1)))
    plt.show()
