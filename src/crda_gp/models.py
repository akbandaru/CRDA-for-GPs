import gpytorch
from gpytorch.kernels import MaternKernel
import torch

class TanimotoMinMaxKernel(gpytorch.kernels.Kernel):

    """TanimotoMinMaxKernel implementation based on the following research paper.
    https://arxiv.org/pdf/2306.14809"""

    has_lengthscale = False # Tanimoto is scale-invariant by construction

    def forward(self, x1, x2, diag=False, **params):
        # L1 norms of each input vector: shape (..., N) and (..., M)
        x1_norm = x1.abs().sum(dim=-1)
        x2_norm = x2.abs().sum(dim=-1)

        eps = 1e-8

        if diag:
            # x1 and x2 assumed aligned row-wise
            l1_dist = (x1 - x2).abs().sum(dim=-1)
            numerator = x1_norm + x2_norm - l1_dist
            denominator = x1_norm + x2_norm + l1_dist
            return numerator / (denominator + eps)
        
        # Pairwise L1 distance
        l1_dist = torch.cdist(x1, x2, p=1) # Shape (..., N, M)
        
        # Broadcast norms to (..., N, M)
        norm_sum = x1_norm.unsqueeze(-1) + x2_norm.unsqueeze(-2)

        numerator = norm_sum - l1_dist
        denominator = norm_sum + l1_dist

        return numerator / (denominator + eps)
    
class ARDExactGPModel_Sparse(gpytorch.models.ExactGP):
    """Gaussian Process Architecture for Spare Datasets (i.e Fingerprint datasets)
    
    Params:
        - train_x: train features
        - train_y: train labels
        - likelihood: likelihood of Gaussian Process  
    """
    def __init__(self, 
                 train_x: torch.Tensor, 
                 train_y: torch.Tensor, 
                 likelihood: gpytorch.likelihoods):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            TanimotoMinMaxKernel() + MaternKernel()
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ARDExactGPModel_Dense(gpytorch.models.ExactGP):
    """Gaussian Process Architecture for Dense Datasets (i.e Composition datasets)
        
        Params:
            - train_x: train features
            - train_y: train labels
            - likelihood: likelihood of Gaussian Process  
        """
    def __init__(self, 
                 train_x: torch.Tensor, 
                 train_y: torch.Tensor, 
                 likelihood: gpytorch.likelihoods):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=train_x.shape[1]
            )
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)