import gpytorch
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from scipy.stats import entropy

def train_step(X_train: torch.Tensor, 
               y_train: torch.Tensor, 
               model: gpytorch.models.ExactGP, 
               likelihood: gpytorch.likelihoods, 
               lr: float=0.1, 
               num_iter: int=100):
    """Train function for training Gaussian Processes
    
    Args:
        X_train: training features
        y_train: training labels
        model: model to fit the data on
        likelihood: likelihood of the Gaussian process
        lr: learning rate hyperparameter for optimizer
        num_iter: number of epochs to train for
    """
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    num_iter = num_iter

    for i in range(num_iter):
        optimizer.zero_grad()

        output = model(X_train)
        loss = -mll(output, y_train)

        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Iter: {i+1}/{num_iter} | Loss: {loss.item():.3f} | Noise: {likelihood.noise.item():.3f}")

def val_step(X_test: torch.Tensor, 
             model: gpytorch.models.ExactGP, 
             likelihood:gpytorch.likelihoods):
    """Validation function for making predictions on the test set
    
    Args:
        - X_test: test features to predict on
        - model: trained model to perform the predictions
        - likelihood: likelihood of the trained Gaussian Process
    
    Returns:
        - pred_mean: the prediction label
    """
    
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # print(f"X_test type: {X_test.dtype}")
        pred_dist = likelihood(model(X_test))

        pred_mean = pred_dist.mean
        pred_lower, pred_upper = pred_dist.confidence_region()

    # print(f"Predictive mean: {pred_mean}")
    # print(f"95% confidence lower: {pred_lower}")
    # print(f"95% confidence upper: {pred_upper}")

    return pred_mean

def set_seed(seed: int = 42):
    """Set the random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = False

def entropy_cal(X: torch.Tensor, bins: int=10):
    """Entropy calculator for Informativity metric
    
    Args:
        - X: the features in dataset
        - bins: number of bins for the histrogram

    Returns:
        - entropies: the entropy calculation per feature in dataset
    """
    X = X.detach().numpy()

    entropies = []

    for i in range(X.shape[1]):
        counts, _ = np.histogramdd(X[:, i], bins=bins)
        p = counts / counts.sum()
        p = p[p > 0]
        entropies.append(entropy(p, base=2).item())
    
    return entropies

def plot_comparsion_entropies(X: torch.Tensor, entropy_1: list[int], entropy_2: list[int], width: float=0.35):
    """Plots a bar graph plot comparing the entropies for each feature.
    
    Args:
        - X: features of the dataset
        - entropy_1: list of entropies for each feature (CRDA)
        - entropy_2: list of entropies for each feature (No CRDA)
        - width: line space width
    """
    fig, ax = plt.subplots()

    x = np.arange(X.shape[-1])
    width = width

    ax.bar(x - width/2, entropy_1, width, label="CRDA")
    ax.bar(x + width/2, entropy_2, width, label="No CRDA")

    ax.axhline(sum(entropy_1) / len(entropy_1))
    ax.axhline(sum(entropy_2) / len(entropy_2))

    print(f"Mean of True vs Aug: {sum(entropy_2) / len(entropy_2)} | {sum(entropy_1) / len(entropy_1)}")

    ax.set_xlabel('Features')
    ax.set_ylabel('Entropy')
    ax.set_title('Grouped Bar Comparsion')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend()

    plt.tight_layout()
    plt.show()