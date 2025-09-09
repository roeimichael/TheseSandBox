"""
models/bayes_logreg_pymc.py
Bayesian Logistic Regression using PyMC with NUTS sampling.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


def fit_bayes_logreg(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    cfg: dict
) -> az.InferenceData:
    """
    Fit Bayesian Logistic Regression using PyMC with NUTS.
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        cfg: Configuration dict with NUTS parameters
        
    Returns:
        InferenceData object with posterior samples
    """
    bayes_cfg = cfg.get("bayes", {}).get("bayes_logreg", {})
    
    draws = bayes_cfg.get("draws", 2000)
    tune = bayes_cfg.get("tune", 1000)
    chains = bayes_cfg.get("chains", 2)
    target_accept = bayes_cfg.get("target_accept", 0.9)
    
    n_features = X_train.shape[1]
    
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal("intercept", mu=0, sigma=5.0)
        beta = pm.Normal("beta", mu=0, sigma=5.0, shape=n_features)
        
        # Linear predictor
        mu = intercept + pm.math.dot(X_train, beta)
        
        # Likelihood
        y = pm.Bernoulli("y", p=pm.math.sigmoid(mu), observed=y_train)
        
        # Sample with NUTS
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=42
        )
    
    # Log diagnostics
    summary = az.summary(idata)
    print(f"Bayesian LR Diagnostics:")
    print(f"R-hat max: {summary['r_hat'].max():.3f}")
    print(f"ESS min: {summary['ess_bulk'].min():.1f}")
    
    # Warn if diagnostics are poor
    if summary['r_hat'].max() > 1.05:
        print("⚠️  Warning: Some R-hat values > 1.05 (poor convergence)")
    if summary['ess_bulk'].min() < 200:
        print("⚠️  Warning: Some ESS values < 200 (poor effective sample size)")
    
    return idata


def predict_proba_members(
    idata: az.InferenceData, 
    X: np.ndarray, 
    cfg: dict
) -> List[np.ndarray]:
    """
    Generate member predictions from posterior samples.
    
    Strategy:
    - Build parameter vectors theta = [intercept, beta...]
    - Select a diverse subset of size max_members using a greedy farthest-point
      strategy in parameter space to encourage variability among members.
    - Fall back to uniform spacing over all draws if needed.
    """
    bayes_cfg = cfg.get("bayes", {}).get("bayes_logreg", {})
    max_members = int(bayes_cfg.get("max_members", 10))

    # Access posterior arrays with chain and draw dimensions
    # Shapes: (chain, draw, ...)
    intercept_da = idata.posterior["intercept"]  # (C, D)
    beta_da = idata.posterior["beta"]            # (C, D, F)

    C = intercept_da.sizes["chain"]
    D = intercept_da.sizes["draw"]
    F = beta_da.sizes["beta_dim"] if "beta_dim" in beta_da.sizes else beta_da.shape[-1]

    # Flatten chains and draws → (N,)
    intercept_flat = intercept_da.values.reshape(C * D)
    beta_flat = beta_da.values.reshape(C * D, F)

    # Parameter matrix Theta: (N, F+1)
    theta = np.concatenate([intercept_flat.reshape(-1, 1), beta_flat], axis=1)

    N = theta.shape[0]
    if N == 0:
        raise ValueError("No posterior draws available to form ensemble members.")

    # Normalize each dimension for fair distance computation
    theta_std = theta.copy()
    col_std = theta_std.std(axis=0)
    col_std[col_std == 0] = 1.0
    theta_std = (theta_std - theta_std.mean(axis=0)) / col_std

    # Greedy farthest-point selection
    selected_indices: List[int] = []
    # Start from middle draw as seed to be deterministic
    seed_idx = N // 2
    selected_indices.append(seed_idx)
    if max_members > 1:
        # Precompute distances to speed up iterative updates
        # We'll keep track of min distance to the selected set for each candidate
        min_dists = np.linalg.norm(theta_std - theta_std[seed_idx], axis=1)
        for _ in range(1, max_members):
            # Exclude already selected indices by setting their distance to -inf
            min_dists[selected_indices] = -np.inf
            # Pick the candidate with the largest min distance to the selected set
            next_idx = int(np.argmax(min_dists))
            if min_dists[next_idx] == -np.inf:
                break
            selected_indices.append(next_idx)
            # Update min distances with the new selected point
            d_new = np.linalg.norm(theta_std - theta_std[next_idx], axis=1)
            min_dists = np.minimum(min_dists, d_new)

    # If we couldn't pick enough (pathological), fall back to uniform spacing
    if len(selected_indices) < max_members:
        fallback = np.linspace(0, N - 1, max_members, dtype=int).tolist()
        selected_indices = fallback

    # Build probabilities for selected members
    member_probas: List[np.ndarray] = []
    for flat_idx in selected_indices:
        cidx = flat_idx // D
        didx = flat_idx % D
        intercept = intercept_da.isel(chain=int(cidx), draw=int(didx)).values
        beta = beta_da.isel(chain=int(cidx), draw=int(didx)).values
        # Linear predictor
        mu = intercept + np.dot(X, beta)
        p = 1 / (1 + np.exp(-mu))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        proba = np.vstack([1 - p, p]).T
        member_probas.append(proba)

    print(f"Selected {len(member_probas)} diverse ensemble members from posterior draws")
    return member_probas


def average_proba(proba_list: List[np.ndarray]) -> np.ndarray:
    """
    Average probabilities across ensemble members.
    
    Args:
        proba_list: List of probability arrays, each (n_samples, n_classes)
        
    Returns:
        Averaged probabilities of shape (n_samples, n_classes)
    """
    if not proba_list:
        raise ValueError("proba_list cannot be empty")
    
    # Stack along new axis and take mean
    stacked = np.stack(proba_list, axis=0)
    return np.mean(stacked, axis=0)
