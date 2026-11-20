import gpytorch
import numpy as np
import torch


def CRDA_function(X_train: torch.Tensor, 
                  y_train: torch.Tensor, 
                  model: gpytorch.models.ExactGP, 
                  likelihood: gpytorch.likelihoods, 
                  perturb_idx: list[int]=[1, 3], 
                  p:float=0.1, 
                  M:int=1):
    """CRDA function implementation based on the following paper: https://openreview.net/pdf?id=dVCOrh99ZB
    
    Args:
        - X_train: train features
        - y_train: train labels
        - model: Gaussian Process model to perform predictions
        - likelihood: likelihood of the Gaussian Process
        - perturb_idx: the features that need to changed
        - p: the amount to change perturb_idx features
        - M: number of syntheic sample(s) per real sample
    
    Returns:
        - x_final: augmented features of the dataset 
        - y_final: augmented labels of the dataset
    """

    # Compute the residual noise
    np.random.seed(42)
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(X_train))

        pred_mean = pred_dist.mean

    residual_noise = y_train - pred_mean

    X_aug, y_aug = [], []

    print("Starting the Loop")
    for i in range(len(X_train)):
        for _ in range(M):
            print(f"Iter: {i}")
            x_new = X_train[i].clone()
            # perturb selected features
            delta = np.random.uniform(-p, p)
            x_new[perturb_idx] *= (1 + delta)
            x_new = x_new.unsqueeze(0)
            
            # new prediction
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_dist = likelihood(model(x_new))

                y_new = pred_dist.mean

            # add original residual
            y_new += residual_noise[i]

            X_aug.append(x_new.squeeze())
            y_aug.append(y_new.squeeze())
    
    X_aug = torch.stack(X_aug).float()
    y_aug = torch.stack(y_aug).float()

    # combine original and augmented data
    x_final = torch.vstack([X_train, X_aug])
    y_final = torch.concatenate([y_train, y_aug])

    return x_final, y_final, X_aug, y_aug 


def random_benchmark(X_train: torch.Tensor, 
                     y_train: torch.Tensor, 
                     model: gpytorch.models.ExactGP, 
                     likelihood: gpytorch.likelihoods, 
                     perturb_idx: list[int]=[1, 3], 
                     p: float=0.1, 
                     M: int=1):
    """Random function generation for random benchmarking
    Args:
        - X_train: train features
        - y_train: train labels
        - model: Gaussian Process model to perform predictions
        - likelihood: likelihood of the Gaussian Process
        - perturb_idx: the features that need to changed
        - p: the amount to change perturb_idx features
        - M: number of syntheic sample(s) per real sample
        
    Returns:
        - x_final: augmented features of the dataset 
        - y_final: augmented labels of the dataset
    """
    X_aug, y_aug = [], []
    model.eval()
    print("Starting the Loop")
    for i in range(len(X_train)):
        for _ in range(M):
            print(f"Iter: {i}")
            x_new = X_train[i].clone()
            # print(f"x_new shape: {x_new.shape}")
            # perturb selected features
            delta = np.random.uniform(-p, p)
            x_new[perturb_idx] *= (1 + delta)
            x_new = x_new.unsqueeze(0)
            # print(f"x_new shape: {x_new.shape}")
            # new prediction
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_dist = likelihood(model(x_new))

                y_new = pred_dist.mean

            X_aug.append(x_new.squeeze())
            y_aug.append(y_new.squeeze())
    
    X_aug = torch.stack(X_aug).float()
    y_aug = torch.stack(y_aug).float()

    # print(f"X_aug shape: {X_aug.shape}")
    # print(f"X_train shape: {X_train.shape}")

    # print(f"y_aug shape: {y_aug.shape}")
    # print(f"y_train shape: {y_train.shape}")

    # combine original and augmented data
    x_final = torch.vstack([X_train, X_aug])
    y_final = torch.concatenate([y_train, y_aug])

    return x_final, y_final 