"""Conditional BLUP simulations of within-person change (published Q2 criterion).

The model is fitted to the full longitudinal dataset; *conditioning* of each
participant's random effects uses only visits before the endpoint. This is not
an out-of-sample prediction or an independent validation procedure.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import truncnorm


@dataclass
class ChangeSimulationResult:
    summary: pd.DataFrame
    changes: pd.DataFrame


def _draw_mvn(rng, mean, covariance, size=None):
    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.min() < -1e-7 * max(1.0, abs(eigenvalues).max()):
        raise ValueError("Estimated covariance matrix is not positive semidefinite")
    transform = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    shape = (len(mean),) if size is None else (size, len(mean))
    return np.asarray(mean) + rng.standard_normal(shape) @ transform.T


def _draw_outcome(rng, mean, sd, bounds, round_to_int):
    low, high = bounds
    if low is not None and high is not None and low >= high:
        raise ValueError("Lower bound must be below upper bound")
    if sd <= 0 or not np.isfinite(sd):
        raise ValueError("Residual standard deviation must be positive")
    if low is None and high is None:
        value = rng.normal(mean, sd)
    else:
        a = -np.inf if low is None else (low - mean) / sd
        b = np.inf if high is None else (high - mean) / sd
        value = truncnorm.rvs(a, b, loc=mean, scale=sd, random_state=rng)
    return float(np.rint(value) if round_to_int else value)


def conditional_random_effects(model, frame, pre_idx, outcome):
    """BLUP mean and conditional covariance using selected row positions.

    `frame` must be aligned to the fitted model, with a zero-based row index.
    The model may have been fitted to all visits, but only `pre_idx` enters
    the participant-specific conditioning calculation.
    """
    X = np.asarray(model.model.exog, dtype=float)
    Z = np.asarray(model.model.exog_re, dtype=float)
    G = np.asarray(model.cov_re, dtype=float)
    beta = np.asarray(model.fe_params, dtype=float)
    sigma2 = float(model.scale)
    pre_idx = np.asarray(pre_idx, dtype=int)
    if not len(pre_idx):
        return np.zeros(G.shape[0]), G.copy()
    Xp, Zp = X[pre_idx], Z[pre_idx]
    yp = frame.iloc[pre_idx][outcome].to_numpy(float)
    V = Zp @ G @ Zp.T + sigma2 * np.eye(len(pre_idx))
    factor = cho_factor(V, lower=True, check_finite=True)
    gain = cho_solve(factor, Zp @ G).T
    mean = gain @ (yp - Xp @ beta)
    covariance = G - gain @ Zp @ G
    return mean, (covariance + covariance.T) / 2


def simulate_individual_change(
    model, data, *, outcome, id_col="participant_id", visit_col="timepoint",
    baseline_visit="T0", endpoint_visit="T4", n_simulations=1000,
    improvement_probability=0.84, bounds=(None, None), round_to_int=False,
    direction="increase", seed=42, min_visits=3,
):
    """Simulate baseline-to-endpoint change and classify P(improvement) >= threshold.

    `data` MUST be the aligned `ModelSelectionResult.data`, not the original
    unfiltered CSV. Visits earlier than endpoint (excluding the endpoint) are
    used to condition the random effects. The full model's fitted parameters
    are used, including its fixed-effect covariance and residual variance.
    """
    if not isinstance(n_simulations, int) or n_simulations < 1:
        raise ValueError("n_simulations must be a positive integer")
    if not 0 <= improvement_probability <= 1:
        raise ValueError("improvement_probability must be between 0 and 1")
    if direction not in {"increase", "decrease"}:
        raise ValueError("direction must be 'increase' or 'decrease'")
    for col in (outcome, id_col, visit_col):
        if col not in data.columns:
            raise KeyError(f"Missing column: {col}")
    if len(data) != len(model.model.endog):
        raise ValueError("data rows must match the fitted model's rows")
    if not np.allclose(data[outcome].to_numpy(float), model.model.endog, equal_nan=True):
        raise ValueError("data rows/order do not match the fitted model; pass selection.data")

    # The original implementation uses Tj labels; accept integer or string labels.
    frame = data.reset_index(drop=True).copy()
    if frame.duplicated([id_col, visit_col]).any():
        raise ValueError("Duplicate participant/visit combinations are not supported")
    if pd.api.types.is_numeric_dtype(frame[visit_col]):
        baseline_visit = float(baseline_visit)
        endpoint_visit = float(endpoint_visit)
    visits = frame.groupby(id_col)[visit_col].agg(list)
    eligible = [pid for pid, vs in visits.items()
                if baseline_visit in vs and endpoint_visit in vs and len(vs) >= min_visits]
    if not eligible:
        raise ValueError("No participants have baseline, endpoint and sufficient visits")

    X = np.asarray(model.model.exog, float)
    Z = np.asarray(model.model.exog_re, float)
    beta = np.asarray(model.fe_params, float)
    fe_names = list(model.fe_params.index)
    beta_cov = np.asarray(model.cov_params().loc[fe_names, fe_names], float)
    G = np.asarray(model.cov_re, float)
    residual_variance = float(model.scale)
    if residual_variance <= 0 or not np.isfinite(residual_variance):
        raise ValueError("Invalid fitted residual variance")
    sd = np.sqrt(residual_variance)
    rng = np.random.default_rng(seed)

    # Conditional random-effects posterior, shared with the validation harness.
    posteriors = {}
    positions = {}
    for pid in eligible:
        idx = np.flatnonzero(frame[id_col].to_numpy() == pid)
        visit_values = frame.iloc[idx][visit_col]
        endpoint_idx = idx[np.asarray(visit_values == endpoint_visit)][0]
        pre_idx = idx[np.asarray(frame.iloc[idx]["log1p_time_c"] < frame.iloc[endpoint_idx]["log1p_time_c"])]
        posteriors[pid] = conditional_random_effects(model, frame, pre_idx, outcome)
        positions[pid] = (idx[np.asarray(visit_values == baseline_visit)][0], endpoint_idx)

    beta_draws = _draw_mvn(rng, beta, beta_cov, size=n_simulations)
    records = []
    summaries = []
    for pid in eligible:
        mean, covariance = posteriors[pid]
        random_draws = _draw_mvn(rng, mean, covariance, size=n_simulations)
        i0, i1 = positions[pid]
        deltas = np.empty(n_simulations)
        for s in range(n_simulations):
            mu0 = X[i0] @ beta_draws[s] + Z[i0] @ random_draws[s]
            mu1 = X[i1] @ beta_draws[s] + Z[i1] @ random_draws[s]
            y0 = _draw_outcome(rng, mu0, sd, bounds, round_to_int)
            y1 = _draw_outcome(rng, mu1, sd, bounds, round_to_int)
            deltas[s] = y1 - y0
        better = deltas > 0 if direction == "increase" else deltas < 0
        probability = float(np.mean(better))
        summaries.append({id_col: pid, "n_pre_endpoint_visits": int(sum(frame.iloc[np.flatnonzero(frame[id_col].to_numpy() == pid)]["log1p_time_c"] < frame.iloc[i1]["log1p_time_c"])),
                          "observed_change": float(frame.iloc[i1][outcome] - frame.iloc[i0][outcome]),
                          "probability_improvement": probability,
                          "improved": bool(probability >= improvement_probability),
                          "simulated_change_median": float(np.median(deltas)),
                          "simulated_change_p025": float(np.quantile(deltas, .025)),
                          "simulated_change_p975": float(np.quantile(deltas, .975))})
        records.extend(({id_col: pid, "simulation": s + 1, "change": float(d)}
                        for s, d in enumerate(deltas)))
    return ChangeSimulationResult(pd.DataFrame(summaries), pd.DataFrame(records))
