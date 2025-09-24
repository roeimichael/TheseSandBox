# models/bayes_mlp_pyro.py
"""
Bayesian MLP using Pyro with SVI (AutoNormal guide) and posterior sampling via Predictive.
Provides a similar API to the LR module but returns a lightweight Posterior object
instead of ArviZ InferenceData. Selection and proba helpers mirror the LR flow.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable
import numpy as np
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
from sklearn.metrics import accuracy_score
from evaluation.helpers import conditional_prediction_correlation as conditional_correlation
from evaluation.metrics import expected_calibration_error as _ece


def _to_tensor(x, device: str, dtype=torch.float32) -> torch.Tensor:
    if hasattr(x, "to_numpy"):
        arr = x.to_numpy()
    elif isinstance(x, (np.ndarray, list, tuple)):
        arr = np.asarray(x)
    else:
        arr = np.array(x)
    return torch.as_tensor(arr, dtype=dtype, device=device)


def _get_cfg(cfg: dict, strategy: dict | None = None) -> dict:
    """Extract effective bayesian MLP config.

    Merges:
      - cfg['bayes']['bayes_mlp'] (mandatory section)
      - any sibling keys inside cfg['bayes'] (e.g. trajectory controls: mode, total_svi_steps, ...)
      - strategy overrides (per-experiment modifications)
    """
    bayes_section = cfg.get("bayes", {}) or {}
    base = (bayes_section.get("bayes_mlp", {}) or {}).copy()
    # absorb sibling keys (trajectory parameters live here)
    for k, v in bayes_section.items():
        if k == "bayes_mlp":
            continue
        # do not overwrite explicit model hyperparams if duplicated
        if k not in base:
            base[k] = v
    if strategy:
        base.update(strategy)
    required = [
        "hidden_sizes", "prior_scale", "random_seed", "svi_steps", "lr", "batch_size",
        "posterior_samples", "max_members", "candidate_pool_size", "sample_strategy", "thinning_step"
    ]
    missing = [k for k in required if k not in base]
    if missing:
        raise KeyError(
            f"Missing bayes_mlp config keys: {missing}. Provide them in ensemble_mlp_config.yaml"
        )
    return base


def _mlp_forward(weights: Dict[str, torch.Tensor], X: torch.Tensor) -> torch.Tensor:
    """Forward pass using explicit weight/bias tensors in `weights` dict.
    Expects keys: W0,b0, W1,b1, ..., with last layer producing 1 logit.
    """
    h = X
    l = 0
    while f"W{l}" in weights:
        W = weights[f"W{l}"]
        b = weights[f"b{l}"]
        h = h @ W + b
        if f"W{l+1}" in weights:
            h = F.relu(h)
        l += 1
    return h.squeeze(-1)  # (n,)


def _build_prior_params(input_dim: int, hidden_sizes: List[int]) -> List[Tuple[int, int]]:
    sizes = []
    prev = input_dim
    for hs in hidden_sizes:
        sizes.append((prev, hs))
        prev = hs
    sizes.append((prev, 1))
    return sizes


def mlp_model(X: torch.Tensor, y: torch.Tensor, prior_scale: float, hidden_sizes: List[int]):
    device = X.device
    layer_sizes = _build_prior_params(X.shape[1], hidden_sizes)
    weights: Dict[str, torch.Tensor] = {}
    for l, (din, dout) in enumerate(layer_sizes):
        W = pyro.sample(
            f"W{l}",
            dist.Normal(torch.zeros(din, dout, device=device), prior_scale * torch.ones(din, dout, device=device)).to_event(2),
        )
        b = pyro.sample(
            f"b{l}",
            dist.Normal(torch.zeros(dout, device=device), prior_scale * torch.ones(dout, device=device)).to_event(1),
        )
        weights[f"W{l}"] = W
        weights[f"b{l}"] = b
    logits = _mlp_forward(weights, X)
    # Plate over observations to match batched likelihood shape (supports minibatching)
    with pyro.plate("data", X.shape[0]):
        pyro.sample("y", dist.Bernoulli(logits=logits), obs=y)


@dataclass
class Posterior:
    """Holds variational guide and pre-sampled weights for reuse.
    samples: dict mapping param name -> tensor (K, shape)
    cfg: effective bayes_mlp config
    device: torch device string
    """
    guide: AutoNormal
    samples: Dict[str, torch.Tensor]
    cfg: dict
    device: str


def _weight_site_names(input_dim: int, hidden_sizes: List[int]) -> List[str]:
    layer_sizes = _build_prior_params(input_dim, hidden_sizes)
    names: List[str] = []
    for l in range(len(layer_sizes)):
        names.extend([f"W{l}", f"b{l}"])
    return names


def _sample_weight_sites(model_fn, guide, X_t: torch.Tensor, y_t: torch.Tensor, site_names: Iterable[str], num_samples: int) -> Dict[str, torch.Tensor]:
    predictive = Predictive(model_fn, guide=guide, num_samples=num_samples, return_sites=list(site_names))
    sampled = predictive(X_t, y_t)
    return {k: v.detach() for k, v in sampled.items() if k in site_names}


def _train_svi(svi: SVI, X_t: torch.Tensor, y_t: torch.Tensor, n_steps: int, batch_size: int) -> None:
    n = X_t.shape[0]
    for step in range(1, n_steps + 1):
        if batch_size >= n:
            loss = svi.step(X_t, y_t)
        else:
            idx = torch.randint(0, n, (batch_size,), device=X_t.device)
            loss = svi.step(X_t[idx], y_t[idx])
        if step % max(1, n_steps // 10) == 0:
            print(f"[SVI] step {step}/{n_steps} loss={loss:.2f}")


def _train_with_trajectory(svi: SVI, guide, model_fn, X_t: torch.Tensor, y_t: torch.Tensor, cfg: dict) -> Dict[str, torch.Tensor]:
    total_steps = int(cfg.get("total_svi_steps", cfg.get("svi_steps", 0)))
    start = int(cfg.get("trajectory_start", 0))
    interval = int(cfg.get("trajectory_interval", 100)) or 100
    max_models = int(cfg.get("trajectory_max_models", 0)) or 0
    batch_size = int(cfg["batch_size"]) or X_t.shape[0]
    site_names = _weight_site_names(X_t.shape[1], list(cfg["hidden_sizes"]))
    trajectory: Dict[str, List[torch.Tensor]] = {}
    collected = 0
    n = X_t.shape[0]
    for step in range(1, total_steps + 1):
        if batch_size >= n:
            loss = svi.step(X_t, y_t)
        else:
            idx = torch.randint(0, n, (batch_size,), device=X_t.device)
            loss = svi.step(X_t[idx], y_t[idx])
        if step % max(1, total_steps // 10) == 0:
            print(f"[SVI] step {step}/{total_steps} loss={loss:.2f}")
        if step >= start and (step - start) % interval == 0 and collected < max_models:
            with torch.no_grad():
                sampled = _sample_weight_sites(model_fn, guide, X_t, y_t, site_names, num_samples=1)
            for k, v in sampled.items():
                if v.dim() == 3 and v.size(0) == 1:
                    # (1, din, dout) stays; if (1,dout) ok
                    pass
                elif v.dim() in (1, 2):
                    v = v.unsqueeze(0)
                trajectory.setdefault(k, []).append(v)
            collected += 1
            print(f"[Trajectory] Collected {collected}/{max_models} at step {step}")
        if collected >= max_models and max_models > 0:
            # Still finish training but skip further collection
            continue
    if not trajectory:
        return {}
    stacked = {k: torch.cat(v, dim=0) for k, v in trajectory.items()}
    print(f"[BNN] Trajectory mode: collected {next(iter(stacked.values())).shape[0]} checkpoints")
    return stacked


def fit_bayes_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    strategy: dict | None = None,
) -> Posterior:
    bayes = _get_cfg(cfg, strategy)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = int(bayes.get("random_seed", 0))
    np.random.seed(rng)
    torch.manual_seed(rng)
    pyro.set_rng_seed(rng)

    X_t = _to_tensor(X_train, device)
    y_t = _to_tensor(y_train, device)

    # Pre-compute class weighting if enabled
    class_weighting = str(bayes.get("class_weighting", "none")).lower()
    custom_weights = bayes.get("class_weights", None)
    if class_weighting == "balanced":
        pos_frac = float(y_t.mean().item())
        neg_frac = 1.0 - pos_frac
        # Balanced scheme: w_pos = 0.5/pos_frac, w_neg = 0.5/neg_frac
        w_pos = 0.5 / max(pos_frac, 1e-6)
        w_neg = 0.5 / max(neg_frac, 1e-6)
    elif class_weighting == "custom" and custom_weights:
        w_neg, w_pos = float(custom_weights[0]), float(custom_weights[1])
    else:
        w_pos, w_neg = 1.0, 1.0

    def model_fn(X, y):
        # Inline model to allow weighting adjustment
        device_local = X.device
        layer_sizes = _build_prior_params(X.shape[1], list(bayes["hidden_sizes"]))
        weights_local: Dict[str, torch.Tensor] = {}
        for l, (din, dout) in enumerate(layer_sizes):
            W = pyro.sample(
                f"W{l}",
                dist.Normal(torch.zeros(din, dout, device=device_local), float(bayes["prior_scale"]) * torch.ones(din, dout, device=device_local)).to_event(2),
            )
            b = pyro.sample(
                f"b{l}",
                dist.Normal(torch.zeros(dout, device=device_local), float(bayes["prior_scale"]) * torch.ones(dout, device=device_local)).to_event(1),
            )
            weights_local[f"W{l}"] = W; weights_local[f"b{l}"] = b
        logits_local = _mlp_forward(weights_local, X)
        with pyro.plate("data", X.shape[0]):
            pyro.sample("y", dist.Bernoulli(logits=logits_local), obs=y)
        # Apply weighting correction via pyro.factor (difference between weighted and unweighted log-likelihood)
        if class_weighting in ("balanced", "custom"):
            # Compute base and weighted log-prob components
            p = torch.sigmoid(logits_local).clamp(1e-6, 1 - 1e-6)
            base_logp = y * torch.log(p) + (1 - y) * torch.log(1 - p)
            weighted_logp = w_pos * y * torch.log(p) + w_neg * (1 - y) * torch.log(1 - p)
            correction = (weighted_logp - base_logp).sum()
            pyro.factor("class_weight_adjust", correction)
        return None

    guide = AutoNormal(model_fn)
    svi = SVI(model_fn, guide, ClippedAdam({"lr": float(bayes["lr"])}), loss=Trace_ELBO())

    mode = str(bayes.get("mode", "posterior")).lower()
    if mode == "trajectory":
        kept = _train_with_trajectory(svi, guide, model_fn, X_t, y_t, bayes)
        if not kept:
            print("[BNN] Trajectory config produced no checkpoints; falling back to posterior sampling.")
    else:
        _train_svi(svi, X_t, y_t, int(bayes["svi_steps"]), int(bayes["batch_size"]))
        kept = {}

    if not kept:  # posterior sampling path
        K = int(bayes["posterior_samples"]) or 200
        site_names = _weight_site_names(X_t.shape[1], list(bayes["hidden_sizes"]))
        kept = _sample_weight_sites(model_fn, guide, X_t, y_t, site_names, num_samples=K)
        print(f"[BNN] Collected {next(iter(kept.values())).shape[0]} posterior samples (device={device})")
    return Posterior(guide=guide, samples=kept, cfg=bayes, device=device)


def _vectorized_logits(samples: Dict[str, torch.Tensor], X: np.ndarray, device: str, batch: int = 256) -> torch.Tensor:
    """Compute logits for all samples in a vectorized (batched) way.
    Returns logits tensor of shape (K, n)
    """
    X_t = _to_tensor(X, device)
    K = next(iter(samples.values())).shape[0]
    n = X_t.shape[0]
    out = torch.empty((K, n), device=device, dtype=X_t.dtype)
    L = 0
    # Determine number of layers by scanning keys
    while f"W{L}" in samples:
        L += 1
    # Process in batches over samples to save memory
    start = 0
    while start < K:
        end = min(start + batch, K)
        B = end - start
        # Build batched input once: (B, n, din)
        h = X_t.unsqueeze(0).expand(B, -1, -1)
        for l in range(L):
            W = samples[f"W{l}"][start:end]  # (B, din, dout) possibly with extra singleton dims
            b = samples[f"b{l}"][start:end]  # (B, dout) possibly with extra singleton dims
            # Normalize shapes: remove any extra singleton dimensions that sometimes appear from Predictive
            if W.dim() == 4 and W.size(1) == 1:
                W = W.squeeze(1)
            if b.dim() == 3 and b.size(1) == 1:
                b = b.squeeze(1)
            if W.dim() == 2:
                W = W.unsqueeze(0).expand(B, -1, -1)
            if b.dim() == 1:
                b = b.unsqueeze(0).expand(B, -1)
            # Robust batched matmul using einsum to avoid broadcasting pitfalls
            # h: (B,n,din), W: (B,din,dout) -> (B,n,dout)
            h = torch.einsum('bnd,bdh->bnh', h, W)
            h = h + b[:, None, :]
            if l < L - 1:
                h = torch.relu(h)
        if h.shape[-1] != 1:
            raise RuntimeError(f"Expected final layer to produce 1 logit, got shape {tuple(h.shape)}")
        out[start:end] = h[..., 0]  # (B,n)
        start = end
    return out  # (K, n)


def select_member_indices(
    posterior: Posterior,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    strategy: dict | None = None,
) -> List[int]:
    """Unified risk-based selection (simplified).

    Process:
      1. Build candidate pool from posterior samples (thinning / random / linspace).
      2. Compute validation predictions + base quality metric (F1 or accuracy blended with AUC if requested).
      3. Iteratively add members by maximizing a risk score that rewards quality & disagreement
         while penalizing error correlation, probability correlation, ECE and log-loss.

    Removed: legacy basic and multi-mode branches (always uses risk logic). Clustering is
    omitted for simplicity; diversity pressure comes from the correlation / disagreement terms.
    """
    bayes = posterior.cfg.copy()
    if strategy:
        bayes.update(strategy)
    rng = bayes.get("random_seed")
    if rng is not None:
        np.random.seed(int(rng))

    K = next(iter(posterior.samples.values())).shape[0]
    candidate_pool_size = int(bayes.get("candidate_pool_size", 200))
    max_members = int(bayes.get("max_members", 10))
    sample_strategy = str(bayes.get("sample_strategy", "linspace")).lower()
    thinning_step = int(bayes.get("thinning_step", 1)) or 1

    # Candidate indices
    if candidate_pool_size >= K:
        candidate = np.arange(K, dtype=int)
    else:
        if sample_strategy == "random":
            candidate = np.random.choice(K, size=candidate_pool_size, replace=False)
        elif sample_strategy == "thinning":
            stride = max(1, thinning_step)
            candidate = np.arange(0, K, stride, dtype=int)
            if len(candidate) < candidate_pool_size:
                extra = np.setdiff1d(np.linspace(0, K - 1, candidate_pool_size, dtype=int), candidate)
                candidate = np.concatenate([candidate, extra[: candidate_pool_size - len(candidate)]])
            candidate = candidate[:candidate_pool_size]
        else:  # linspace
            candidate = np.linspace(0, K - 1, candidate_pool_size, dtype=int)

    # Validation scoring
    logits = _vectorized_logits(posterior.samples, X_val, posterior.device)
    logits_c = logits[candidate]
    p = torch.sigmoid(logits_c).cpu().numpy()
    p = np.clip(p, 1e-7, 1 - 1e-7)
    proba_val = np.stack([1 - p, p], axis=-1)  # (C,n,2)
    preds = (p >= 0.5).astype(np.int8)

    yv = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.asarray(y_val)
    acc = (preds == yv[None, :]).mean(axis=1)
    tp = (preds & (yv[None, :] == 1)).sum(axis=1)
    fp = (preds & (yv[None, :] == 0)).sum(axis=1)
    fn = ((1 - preds) & (yv[None, :] == 1)).sum(axis=1)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)

    # Metric selection / blend
    selection_metric = str(bayes.get("selection_metric", "f1")).lower()
    alpha_auc = float(bayes.get("selection_alpha_auc", 0.0))
    try:
        from sklearn.metrics import roc_auc_score as _auc
        auc_vals = np.array([
            _auc(yv, proba_val[i, :, 1]) if len(np.unique(yv)) == 2 else 0.5 for i in range(proba_val.shape[0])
        ])
    except Exception:
        auc_vals = np.full((proba_val.shape[0],), 0.5)
    norm_auc = np.clip((auc_vals - 0.5) / 0.5, 0, 1)
    quality_raw = f1 if selection_metric == "f1" else acc
    if np.all(quality_raw == 0):
        quality_raw = acc
    if alpha_auc > 0:
        quality_raw = (1 - alpha_auc) * quality_raw + alpha_auc * norm_auc

    # Risk weights
    risk_weights = bayes.get("risk_weights", {}) or {}
    w_quality = float(risk_weights.get("quality", 1.0))
    w_err = float(risk_weights.get("error_corr", 0.5))
    w_proba = float(risk_weights.get("proba_corr", 0.3))
    w_dis = float(risk_weights.get("disagreement", 0.2))
    w_ece = float(risk_weights.get("ece", 0.1))
    w_ll = float(risk_weights.get("log_loss", 0.1))
    risk_min_gain = float(bayes.get("risk_min_gain", -1e9))

    # Per-candidate log loss & ECE
    pos_probs = proba_val[:, :, 1]
    y_float = yv.astype(float)
    ll_raw = -(y_float[None, :] * np.log(pos_probs) + (1 - y_float[None, :]) * np.log(1 - pos_probs)).mean(axis=1)
    ece_vals = np.array([_ece(yv, proba_val[i]) for i in range(proba_val.shape[0])])
    def _minmax(x):
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-9:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    ll_norm = _minmax(ll_raw)
    ece_norm = _minmax(ece_vals)

    # Correlation (probability) matrix
    if pos_probs.shape[0] > 1:
        corr_matrix = np.corrcoef(pos_probs)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    else:
        corr_matrix = np.ones((1, 1))

    selected_local: List[int] = []
    collapse_penalty = (preds.std(axis=1) == 0).astype(float)
    seed_idx = int(np.argmax(w_quality * quality_raw - 0.5 * collapse_penalty - w_ece * ece_norm - w_ll * ll_norm))
    selected_local.append(seed_idx)

    def risk_score(i: int, current: List[int]) -> float:
        quality = quality_raw[i]
        if not current:
            return w_quality * quality - w_ece * ece_norm[i] - w_ll * ll_norm[i]
        err_corrs = []
        prob_corrs = []
        disagreements = []
        for s in current:
            if preds[i].std() == 0 or preds[s].std() == 0:
                ec = 1.0
            else:
                ec = conditional_correlation(preds[i], preds[s], yv)
                if np.isnan(ec):
                    ec = 0.0
            err_corrs.append(ec)
            pc = corr_matrix[i, s]
            prob_corrs.append(pc)
            disagreements.append(1.0 - (preds[i] == preds[s]).mean())
        err_avg = float(np.mean(err_corrs)) if err_corrs else 0.0
        pc_avg = float(np.mean(prob_corrs)) if prob_corrs else 0.0
        dis_avg = float(np.mean(disagreements)) if disagreements else 0.0
        return float(
            w_quality * quality
            - w_err * max(0.0, err_avg)
            - w_proba * max(0.0, pc_avg)
            + w_dis * dis_avg
            - w_ece * ece_norm[i]
            - w_ll * ll_norm[i]
        )

    while len(selected_local) < min(max_members, len(candidate)):
        best_idx, best_score = None, None
        for i in range(len(candidate)):
            if i in selected_local:
                continue
            score = risk_score(i, selected_local)
            if (best_score is None) or (score > best_score):
                best_score = score; best_idx = i
        if best_idx is None or (best_score is not None and best_score < risk_min_gain):
            break
        selected_local.append(best_idx)

    selected = [int(candidate[i]) for i in selected_local]
    print(f"[BNN] Selected {len(selected)} ensemble members (risk-based) strategy={sample_strategy}")
    return selected


def proba_for_indices(
    posterior: Posterior,
    X: np.ndarray,
    indices: List[int],
) -> List[np.ndarray]:
    logits = _vectorized_logits(posterior.samples, X, posterior.device)
    logits_sel = logits[indices]
    p = torch.sigmoid(logits_sel).cpu().numpy()
    p = np.clip(p, 1e-7, 1 - 1e-7)
    proba = [np.stack([1 - p[i], p[i]], axis=-1) for i in range(p.shape[0])]
    return proba


def average_proba(proba_list: List[np.ndarray]) -> np.ndarray:
    if not proba_list:
        raise ValueError("proba_list cannot be empty")
    return np.mean(np.stack(proba_list, axis=0), axis=0)
