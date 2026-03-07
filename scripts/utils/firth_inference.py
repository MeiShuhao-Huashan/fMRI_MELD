from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


# For df=1, chi2_{0.95} = (z_{0.975})^2
CHI2_0_95_DF1 = 3.841458820694124


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _penalized_loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    eta = X @ beta
    mu = _sigmoid(eta)
    eps = 1e-12
    ll = float(np.sum(y * np.log(mu + eps) + (1.0 - y) * np.log(1.0 - mu + eps)))
    w = mu * (1.0 - mu)
    I = X.T @ (w[:, None] * X)
    # Stabilise determinant (should be PD, but can be near-singular)
    I = I + np.eye(I.shape[0]) * 1e-12
    sign, logdet = np.linalg.slogdet(I)
    if sign <= 0:
        return -math.inf
    return ll + 0.5 * float(logdet)


def firth_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
    init_beta: Optional[np.ndarray] = None,
    fixed: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, float, int, bool]:
    """
    Firth penalized logistic regression (Jeffreys prior), Newton updates with step-halving.

    When `fixed` is provided, those coefficient indices are held constant while the
    remaining coefficients are optimized (used for profile penalized likelihood CI).
    Returns (beta_hat, penalized_loglik, n_iter, converged).
    """
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("X and y length mismatch")

    fixed = fixed or {}
    fixed_idx = set(fixed.keys())
    free_idx = [i for i in range(p) if i not in fixed_idx]

    if init_beta is None:
        beta = np.zeros(p, dtype=float)
    else:
        beta = np.asarray(init_beta, dtype=float).copy()
        if beta.shape != (p,):
            raise ValueError("init_beta has wrong shape")

    for k, v in fixed.items():
        beta[int(k)] = float(v)

    prev_obj = _penalized_loglik(beta, X, y)
    if not math.isfinite(prev_obj):
        prev_obj = -math.inf

    if not free_idx:
        return beta, prev_obj, 0, True

    for it in range(1, max_iter + 1):
        eta = X @ beta
        mu = _sigmoid(eta)
        w = mu * (1.0 - mu)
        w = np.clip(w, 1e-12, None)

        I = X.T @ (w[:, None] * X)
        I = I + np.eye(p) * 1e-9
        try:
            I_inv = np.linalg.inv(I)
        except np.linalg.LinAlgError:
            return beta, -math.inf, it, False

        WX = (np.sqrt(w)[:, None]) * X
        h = np.sum((WX @ I_inv) * WX, axis=1)
        adj = (0.5 - mu) * h
        U = X.T @ (y - mu + adj)

        I_ff = I[np.ix_(free_idx, free_idx)]
        U_f = U[free_idx]
        try:
            delta_f = np.linalg.solve(I_ff, U_f)
        except np.linalg.LinAlgError:
            return beta, -math.inf, it, False

        step = 1.0
        beta_new = beta.copy()
        beta_new[free_idx] = beta[free_idx] + step * delta_f
        for k, v in fixed.items():
            beta_new[int(k)] = float(v)

        obj_new = _penalized_loglik(beta_new, X, y)
        while obj_new < prev_obj and step > 1e-6:
            step *= 0.5
            beta_new[free_idx] = beta[free_idx] + step * delta_f
            for k, v in fixed.items():
                beta_new[int(k)] = float(v)
            obj_new = _penalized_loglik(beta_new, X, y)

        beta = beta_new
        if np.max(np.abs(step * delta_f)) < tol:
            return beta, obj_new, it, True
        prev_obj = obj_new

    return beta, prev_obj, max_iter, False


def chi2_sf_df1(x: float) -> float:
    """
    Survival function for Chi-square(df=1) without SciPy:
      CDF(x) = erf(sqrt(x/2))
      SF(x)  = 1 - CDF(x)
    """
    if not math.isfinite(x) or x < 0:
        return float("nan")
    return float(1.0 - math.erf(math.sqrt(x / 2.0)))


def _profile_lr_stat(
    *,
    X: np.ndarray,
    y: np.ndarray,
    idx: int,
    beta_full: np.ndarray,
    pl_full: float,
    b_fixed: float,
    max_iter: int,
    tol: float,
) -> float:
    beta_init = beta_full.copy()
    beta_init[idx] = float(b_fixed)
    beta_hat, pl_hat, _it, conv = firth_fit(X, y, max_iter=max_iter, tol=tol, init_beta=beta_init, fixed={idx: float(b_fixed)})
    if not conv or not math.isfinite(pl_hat):
        return float("inf")
    lr = 2.0 * (pl_full - pl_hat)
    if not math.isfinite(lr):
        return float("inf")
    return float(max(0.0, lr))


def profile_ci_beta_df1(
    X: np.ndarray,
    y: np.ndarray,
    *,
    idx: int,
    beta_full: np.ndarray,
    pl_full: float,
    alpha: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-8,
    max_bracket_steps: int = 50,
    max_bisect_iter: int = 60,
) -> Tuple[float, float]:
    """
    Profile penalized likelihood CI for a single coefficient (Chi-square df=1 cutoff).
    Returns (ci_lo, ci_hi) on the coefficient scale.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0,1)")

    if abs(alpha - 0.05) > 1e-12:
        raise ValueError("Only alpha=0.05 is supported (df=1) without SciPy.")
    target = CHI2_0_95_DF1
    b_hat = float(beta_full[idx])

    def lr(b: float) -> float:
        return _profile_lr_stat(X=X, y=y, idx=idx, beta_full=beta_full, pl_full=pl_full, b_fixed=b, max_iter=max_iter, tol=tol)

    # ---- lower bound ----
    step = 0.5
    lo = b_hat - step
    lr_lo = lr(lo)
    n_step = 0
    while lr_lo < target and n_step < max_bracket_steps and abs(step) < 64.0:
        step *= 2.0
        lo = b_hat - step
        lr_lo = lr(lo)
        n_step += 1
    if not math.isfinite(lr_lo) or lr_lo < target:
        ci_lo = float("nan")
    else:
        hi = b_hat
        for _ in range(max_bisect_iter):
            mid = 0.5 * (lo + hi)
            lr_mid = lr(mid)
            if not math.isfinite(lr_mid):
                lo = mid
                continue
            if abs(lr_mid - target) < 1e-4:
                lo = mid
                break
            if lr_mid >= target:
                lo = mid
            else:
                hi = mid
        ci_lo = lo

    # ---- upper bound ----
    step = 0.5
    hi = b_hat + step
    lr_hi = lr(hi)
    n_step = 0
    while lr_hi < target and n_step < max_bracket_steps and abs(step) < 64.0:
        step *= 2.0
        hi = b_hat + step
        lr_hi = lr(hi)
        n_step += 1
    if not math.isfinite(lr_hi) or lr_hi < target:
        ci_hi = float("nan")
    else:
        lo2 = b_hat
        for _ in range(max_bisect_iter):
            mid = 0.5 * (lo2 + hi)
            lr_mid = lr(mid)
            if not math.isfinite(lr_mid):
                hi = mid
                continue
            if abs(lr_mid - target) < 1e-4:
                hi = mid
                break
            if lr_mid >= target:
                hi = mid
            else:
                lo2 = mid
        ci_hi = hi

    return float(ci_lo), float(ci_hi)


def firth_or_ci_p(
    X: np.ndarray,
    y: np.ndarray,
    *,
    idx: int = 1,
    alpha: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """
    Convenience wrapper for a single coefficient:
      - OR and profile penalized-likelihood CI
      - p-value from penalized likelihood ratio test (full vs reduced without idx)
    """
    beta_full, pl_full, _it_full, conv_full = firth_fit(X, y, max_iter=max_iter, tol=tol)
    if not conv_full or not math.isfinite(pl_full):
        return {
            "beta": float("nan"),
            "or": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_plr": float("nan"),
            "lr": float("nan"),
        }

    beta = float(beta_full[idx])
    or_ = float(math.exp(beta))

    # PLR test: drop the tested coefficient (nested model)
    X_red = np.delete(X, idx, axis=1)
    beta_red, pl_red, _it_red, conv_red = firth_fit(X_red, y, max_iter=max_iter, tol=tol)
    if (not conv_red) or (not math.isfinite(pl_red)):
        lr = float("nan")
        p_plr = float("nan")
    else:
        lr = float(max(0.0, 2.0 * (pl_full - pl_red)))
        p_plr = chi2_sf_df1(lr)

    ci_b_lo, ci_b_hi = profile_ci_beta_df1(X, y, idx=idx, beta_full=beta_full, pl_full=pl_full, alpha=alpha, max_iter=max_iter, tol=tol)
    ci_lo = float(math.exp(ci_b_lo)) if math.isfinite(ci_b_lo) else float("nan")
    ci_hi = float(math.exp(ci_b_hi)) if math.isfinite(ci_b_hi) else float("nan")

    return {
        "beta": beta,
        "or": or_,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_plr": float(p_plr),
        "lr": lr,
    }


def columns_have_variance(X: np.ndarray, *, idxs: Iterable[int]) -> bool:
    for i in idxs:
        if float(np.std(X[:, int(i)])) == 0.0:
            return False
    return True
