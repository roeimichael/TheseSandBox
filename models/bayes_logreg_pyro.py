# models/bayes_logreg_pyro.py
"""
Bayesian Logistic Regression using Pyro (PyTorch) with NUTS sampling (GPU-capable on Windows).
API-compatible with the previous PyMC version: returns an ArviZ InferenceData with
posterior dimensions ['chain', 'draw', ('beta_dim' for beta)] so downstream code works with minimal changes.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import arviz as az
from sklearn.metrics import accuracy_score
from evaluation.helpers import conditional_prediction_correlation as conditional_correlation


def bayes_logreg_model(X: torch.Tensor, y: torch.Tensor):
    """Top-level Pyro model (picklable) for Windows multiprocessing.
    X: (n_samples, n_features) on device
    y: (n_samples,) on same device
    """
    n_features = X.shape[1]
    loc0 = torch.tensor(0.0, device=X.device)
    scale5 = torch.tensor(5.0, device=X.device)
    intercept = pyro.sample("intercept", dist.Normal(loc0, scale5))
    beta = pyro.sample(
        "beta",
        dist.Normal(loc0, scale5).expand([n_features]).to_event(1)
    )
    logits = intercept + X.matmul(beta)
    pyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

def _to_tensor(x, device: str) -> torch.Tensor:
    """Convert numpy/pandas/list to a float32 tensor on the given device."""
    if hasattr(x, "to_numpy"):
        arr = x.to_numpy()
    elif isinstance(x, (np.ndarray, list, tuple)):
        arr = np.asarray(x)
    else:
        arr = np.array(x)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def fit_bayes_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    strategy: dict | None = None,
) -> az.InferenceData:
    """
    Fit Bayesian Logistic Regression via Pyro NUTS.

    Args:
        X_train: (n_samples, n_features)
        y_train: (n_samples,)
        cfg: config dict (uses cfg['bayes']['bayes_logreg'])

    Returns:
        ArviZ InferenceData with posterior samples for 'intercept' and 'beta'
        shaped (chain, draw, ...) — compatible with your downstream code.
    """
    bayes_cfg = cfg.get("bayes", {}).get("bayes_logreg", {}).copy()
    # Allow overrides from strategy dict (draws/tune/chains/target_accept/thin)
    if strategy:
        for k in ("draws", "tune", "chains", "target_accept", "thin"):
            if k in strategy:
                bayes_cfg[k] = strategy[k]
    draws = int(bayes_cfg.get("draws", 2000))
    tune = int(bayes_cfg.get("tune", 1000))
    chains = int(bayes_cfg.get("chains", 2))
    target_accept = float(bayes_cfg.get("target_accept", 0.9))
    rng_seed = int(bayes_cfg.get("random_seed", 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Seeds for reproducibility
    try:
        np.random.seed(rng_seed)
        torch.manual_seed(rng_seed)
        pyro.set_rng_seed(rng_seed)
    except Exception:
        pass
    X_t = _to_tensor(X_train, device)
    y_t = _to_tensor(y_train, device)

    # Pyro NUTS
    nuts = NUTS(
        bayes_logreg_model,
        target_accept_prob=target_accept,
        max_tree_depth=int(bayes_cfg.get("max_tree_depth", 10)),
        step_size=bayes_cfg.get("step_size", None),
    )
    # On Windows, multiple chains run "sequentially" to avoid fork issues
    # On Windows, avoid multiprocessing pickling issues by defaulting to 1 chain
    chains = max(1, chains)
    mcmc = MCMC(
        nuts,
        num_samples=draws,
        warmup_steps=tune,
        num_chains=chains,
        disable_progbar=False,
    )
    pyro.clear_param_store()
    # y must be float for Bernoulli with logits; clamp to {0,1}
    y_obs = torch.clamp(y_t, 0, 1)
    mcmc.run(X_t, y_obs)

    # Get samples grouped by chain -> shapes: (chains, draws, ...)
    samples = mcmc.get_samples(group_by_chain=True)
    # Move to CPU numpy for ArviZ
    intercept = samples["intercept"].detach().cpu().numpy()  # (C, D)
    beta = samples["beta"].detach().cpu().numpy()            # (C, D, F)

    # Build InferenceData that mimics PyMC dims/names
    idata = az.from_dict(
        posterior={
            "intercept": intercept,         # shape: (chain, draw)
            "beta": beta,                   # shape: (chain, draw, beta_dim)
        },
        coords={"beta_dim": np.arange(beta.shape[-1])},
        dims={"beta": ["beta_dim"], "intercept": []},
    )

    # Diagnostics
    summary = az.summary(idata, var_names=["intercept", "beta"], filter_vars="like")
    rhat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    print("Bayesian LR Diagnostics (Pyro NUTS):")
    print(f"R-hat max: {rhat_max:.3f}")
    print(f"ESS min: {ess_min:.1f}")
    try:
        diagn = mcmc.diagnostics()
        n_div = sum(len(v.get("divergences", [])) for v in diagn.values())
        print(f"[Pyro] NUTS divergences: {n_div}")
    except Exception:
        pass
    if rhat_max > 1.05:
        print("⚠️  Warning: Some R-hat values > 1.05 (poor convergence)")
    if ess_min < 200:
        print("⚠️  Warning: Some ESS values < 200 (low effective sample size)")

    # Small explicit note so you can confirm device
    print(f"[Pyro] Using device: {device} | torch.cuda.is_available()={torch.cuda.is_available()}")
    return idata


# conditional_correlation is imported from evaluation.helpers


# Note: predict_proba_members was removed in favor of select_member_indices + proba_for_indices


def _flatten_posterior(idata: "az.InferenceData") -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """Flatten posterior samples for quick index-based access.
    Returns: intercept_flat (N,), beta_flat (N,F), C, D, F
    """
    intercept_da = idata.posterior["intercept"]  # (C, D)
    beta_da = idata.posterior["beta"]            # (C, D, F)
    C = intercept_da.sizes["chain"]
    D = intercept_da.sizes["draw"]
    F = beta_da.sizes.get("beta_dim", beta_da.shape[-1])
    intercept_flat = intercept_da.values.reshape(C * D)
    beta_flat = beta_da.values.reshape(C * D, F)
    return intercept_flat, beta_flat, C, D, F


def select_member_indices(
    idata: "az.InferenceData",
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    strategy: dict | None = None,
) -> List[int]:
    """Select ensemble member indices once on validation and reuse across splits.
    Strategy and config handling mirrors predict_proba_members.
    Returns a list of flattened posterior indices (chain*draw based).
    """
    bayes_cfg = cfg.get("bayes", {}).get("bayes_logreg", {}).copy()
    if strategy:
        for k in (
            "max_members",
            "candidate_pool_size",
            "sample_strategy",
            "thinning_step",
            "per_chain_candidates",
            "random_seed",
        ):
            if k in strategy:
                bayes_cfg[k] = strategy[k]
    max_members = int(bayes_cfg.get("max_members", 10))
    candidate_pool_size = int(bayes_cfg.get("candidate_pool_size", 50))
    sample_strategy = str(bayes_cfg.get("sample_strategy", "linspace")).lower()
    thinning_step = int(bayes_cfg.get("thinning_step", 100))
    per_chain_candidates = bayes_cfg.get("per_chain_candidates")
    rng_seed = bayes_cfg.get("random_seed")
    if rng_seed is not None:
        np.random.seed(int(rng_seed))

    intercept_flat, beta_flat, C, D, F = _flatten_posterior(idata)
    N = intercept_flat.shape[0]

    Xv_np = X_val.to_numpy() if hasattr(X_val, "to_numpy") else np.asarray(X_val)
    yv_np = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.asarray(y_val)

    def _linspace_candidates():
        return np.linspace(0, N - 1, candidate_pool_size, dtype=int)

    def _random_candidates():
        size = min(candidate_pool_size, N)
        return np.random.choice(N, size=size, replace=False)

    def _per_chain_candidates():
        k = int(per_chain_candidates) if per_chain_candidates is not None else max(1, candidate_pool_size // max(1, C))
        idxs = []
        for c in range(C):
            draws = np.linspace(0, D - 1, num=min(k, D), dtype=int)
            flat = c * D + draws
            idxs.extend(list(flat))
        if len(idxs) < candidate_pool_size:
            remaining = np.setdiff1d(np.arange(N), np.array(idxs), assume_unique=False)
            if len(remaining) > 0:
                top_up = np.random.choice(remaining, size=min(candidate_pool_size - len(idxs), len(remaining)), replace=False)
                idxs.extend(list(top_up))
        return np.array(idxs[:candidate_pool_size], dtype=int)

    def _thinning_candidates():
        if thinning_step <= 0:
            return _linspace_candidates()
        idxs = np.arange(0, N, thinning_step, dtype=int)
        if len(idxs) < candidate_pool_size:
            extra = np.setdiff1d(_linspace_candidates(), idxs)
            pad = extra[: max(0, candidate_pool_size - len(idxs))]
            idxs = np.concatenate([idxs, pad])
        return idxs[:candidate_pool_size]

    if sample_strategy == "random":
        candidate_indices = _random_candidates()
    elif sample_strategy in ("per_chain", "per-chain"):
        candidate_indices = _per_chain_candidates()
    elif sample_strategy in ("thinning", "thin"):
        candidate_indices = _thinning_candidates()
    else:
        candidate_indices = _linspace_candidates()

    # Score candidates on validation
    val_preds = []
    val_accs = []
    for flat_idx in candidate_indices:
        intercept = intercept_flat[flat_idx]
        beta = beta_flat[flat_idx]
        mu_val = intercept + np.dot(Xv_np, beta)
        p_val = 1 / (1 + np.exp(-mu_val))
        p_val = np.clip(p_val, 1e-7, 1 - 1e-7)
        proba_val = np.vstack([1 - p_val, p_val]).T
        pred_val = np.argmax(proba_val, axis=1)
        acc_val = accuracy_score(yv_np, pred_val)
        val_preds.append(pred_val)
        val_accs.append(acc_val)

    # Greedy selection
    max_to_select = min(max_members, len(candidate_indices))
    selected_local = [int(np.argmax(val_accs))]
    while len(selected_local) < max_to_select:
        unselected = [i for i in range(len(candidate_indices)) if i not in selected_local]
        if not unselected:
            break
        min_corr = None
        best_idx = None
        for idx in unselected:
            corrs = [conditional_correlation(val_preds[idx], val_preds[s], yv_np) for s in selected_local]
            avg_corr = np.nanmean(corrs)
            if (min_corr is None) or (avg_corr < min_corr):
                min_corr = avg_corr
                best_idx = idx
        if best_idx is None:
            break
        selected_local.append(best_idx)

    # Map local indices back to flattened posterior indices
    selected_flat = [int(candidate_indices[i]) for i in selected_local]
    print(f"Selected {len(selected_flat)} ensemble members (by indices)")
    return selected_flat


def proba_for_indices(
    idata: "az.InferenceData",
    X: np.ndarray,
    indices: List[int],
) -> List[np.ndarray]:
    """Compute class probabilities for provided flattened posterior indices on inputs X."""
    intercept_flat, beta_flat, *_ = _flatten_posterior(idata)
    X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    member_probas: List[np.ndarray] = []
    for flat_idx in indices:
        intercept = intercept_flat[flat_idx]
        beta = beta_flat[flat_idx]
        mu = intercept + np.dot(X_np, beta)
        p = 1 / (1 + np.exp(-mu))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        proba = np.vstack([1 - p, p]).T
        member_probas.append(proba)
    return member_probas


def average_proba(proba_list: List[np.ndarray]) -> np.ndarray:
    """
    Average probabilities across ensemble members.
    """
    if not proba_list:
        raise ValueError("proba_list cannot be empty")
    return np.mean(np.stack(proba_list, axis=0), axis=0)
