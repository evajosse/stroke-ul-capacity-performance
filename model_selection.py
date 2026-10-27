"""Two-stage longitudinal mixed-effects model selection.

Adapted from the authors' original research scripts. Input is a tidy DataFrame;
no REDCap, IMU, or private configuration dependencies are required.
"""
from dataclasses import dataclass
from pathlib import Path
import logging
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

LOGGER = logging.getLogger("recovery.model_selection")

FIXED_CANDIDATES = {
    "linear_time": "time_c",
    "log_time": "log1p_time_c",
    "quadratic_time": "time_c + time_c2",
    "quadratic_log_time": "log1p_time_c + log1p_time2_c",
}
RANDOM_CANDIDATES = {
    "random_intercept": "1",
    "random_linear_slope": "1 + time_c",
    "random_log_slope": "1 + log1p_time_c",
}


@dataclass
class ModelSelectionResult:
    model: object  # statsmodels MixedLMResults fitted with REML
    data: pd.DataFrame
    random_comparison: pd.DataFrame
    fixed_comparison: pd.DataFrame
    random_formula: str
    fixed_formula: str
    fixed_name: str


def prepare_time_columns(df, time_col, id_col, scale_factor=30.0):
    """Match original time transforms, recalculating columns to avoid stale values.

    Linear: global-mean centered absolute time, divided by scale_factor.
    Log: log1p(time since each participant's first observation / scale_factor),
    then global-mean centered. The original scripts use first *available* time,
    so users should ensure the intended baseline is present.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    for col in (time_col, id_col):
        if col not in df:
            raise KeyError(f"Missing column: {col}")
    out = df.copy()
    out[time_col] = pd.to_numeric(out[time_col], errors="raise")
    if out[time_col].isna().any() or out[id_col].isna().any():
        raise ValueError("Missing time or participant ID")
    t = out[time_col].astype(float)
    out["time_c"] = (t - t.mean()) / scale_factor
    baseline = out.groupby(id_col)[time_col].transform("min")
    relative = (t - baseline) / scale_factor
    out["log1p_time_c"] = np.log1p(relative)
    out["log1p_time_c"] -= out["log1p_time_c"].mean()
    out["time_c2"] = out["time_c"] ** 2
    out["log1p_time2_c"] = out["log1p_time_c"] ** 2
    return out


def _fit(df, outcome, group_col, fixed_formula, random_formula, reml, start=None):
    """Fit with bounded optimizer attempts; preserve diagnostics in a log.

    Non-converged fits, non-finite likelihoods and invalid covariance matrices
    are rejected. A boundary covariance is logged, not automatically rejected:
    it can be a valid ML estimate but warrants interpretation.
    """
    mod = smf.mixedlm(
        f"{outcome} ~ {fixed_formula}", data=df,
        groups=df[group_col], re_formula=random_formula,
    )
    failures = []
    # L-BFGS is generally faster than Nelder-Mead for MixedLM. If it fails,
    # BFGS and conjugate gradient provide independent fallback attempts.
    for method in ("lbfgs", "bfgs", "cg"):
        begin = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = mod.fit(reml=reml, method=method, start_params=start,
                                 maxiter=350, disp=False)
                if not result.converged:
                    raise RuntimeError("optimizer reported non-convergence")
                if not np.isfinite(result.llf):
                    raise RuntimeError("non-finite log-likelihood")
                if not np.all(np.isfinite(np.asarray(result.cov_re))):
                    raise RuntimeError("non-finite random-effects covariance")
                elapsed = time.perf_counter() - begin
                LOGGER.info("FIT OK method=%s reml=%s FE=%s RE=%s seconds=%.2f llf=%.4f warnings=%s",
                            method, reml, fixed_formula, random_formula, elapsed, result.llf,
                            "; ".join(sorted({str(w.message) for w in caught})) or "none")
                return result
            except (ValueError, np.linalg.LinAlgError, RuntimeError, FloatingPointError) as exc:
                failures.append(f"{method}: {exc}")
                LOGGER.warning("FIT FAILED method=%s reml=%s FE=%s RE=%s error=%s warnings=%s",
                               method, reml, fixed_formula, random_formula, exc,
                               "; ".join(sorted({str(w.message) for w in caught})) or "none")
    raise RuntimeError("All optimizers failed: " + " | ".join(failures))


def _reml_aic(result):
    """Original study's REML information criterion for a fixed FE structure.

    Counts random-effect covariance parameters and residual variance only.
    This is intentionally distinct from standard ML AIC.
    """
    q = np.asarray(result.cov_re).shape[0]
    k_variance = q * (q + 1) // 2 + 1
    return -2 * float(result.llf) + 2 * k_variance


def _choose(table, rank_order, aic_tolerance):
    valid = table.loc[table["converged"] & np.isfinite(table["aic"])].copy()
    if valid.empty:
        raise RuntimeError("No converged candidate models with finite AIC")
    minimum = valid["aic"].min()
    eligible = valid.loc[valid["aic"] <= minimum + aic_tolerance].copy()
    eligible["rank"] = eligible["name"].map({name: i for i, name in enumerate(rank_order)})
    return eligible.sort_values(["rank", "aic"]).iloc[0]


def select_mixed_model(df, outcome, time_col="days_post_stroke",
                       group_col="participant_id", scale_factor=30.0,
                       aic_tolerance=4.0):
    """Select random effects by REML-AIC, then fixed effects by ML-AIC.

    Random candidates: intercept, intercept + linear slope, intercept + log slope.
    Fixed candidates (parsimony order): linear, log, quadratic, log quadratic.
    The selected random-effects *formula* remains unchanged in stage two.
    Final selected model is refitted with REML.
    """
    if outcome not in df:
        raise KeyError(f"Missing outcome: {outcome}")
    if aic_tolerance < 0:
        raise ValueError("aic_tolerance must be nonnegative")
    cleaned = df.dropna(subset=[outcome, time_col, group_col]).copy()
    if cleaned.empty or cleaned[group_col].nunique() < 3:
        raise ValueError("At least three participants with nonmissing observations are required")
    data = prepare_time_columns(cleaned, time_col, group_col, scale_factor)
    data = data.reset_index(drop=True)

    re_rows = []
    for name, formula in RANDOM_CANDIDATES.items():
        try:
            res = _fit(data, outcome, group_col, "time_c", formula, reml=True)
            re_rows.append(dict(name=name, formula=formula, aic=_reml_aic(res),
                                converged=True, error=""))
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            re_rows.append(dict(name=name, formula=formula, aic=np.nan,
                                converged=False, error=str(exc)))
    re_table = pd.DataFrame(re_rows)
    selected_re = _choose(re_table, RANDOM_CANDIDATES, aic_tolerance)
    random_formula = selected_re["formula"]

    fe_rows = []
    fe_fits = {}
    for name, formula in FIXED_CANDIDATES.items():
        try:
            res = _fit(data, outcome, group_col, formula, random_formula, reml=False)
            fe_fits[name] = res
            fe_rows.append(dict(name=name, formula=formula, aic=float(res.aic),
                                converged=True, error=""))
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            fe_rows.append(dict(name=name, formula=formula, aic=np.nan,
                                converged=False, error=str(exc)))
    fe_table = pd.DataFrame(fe_rows)
    selected_fe = _choose(fe_table, FIXED_CANDIDATES, aic_tolerance)
    fixed_name = selected_fe["name"]
    final = _fit(data, outcome, group_col, selected_fe["formula"], random_formula,
                 reml=True, start=fe_fits[fixed_name].params)
    return ModelSelectionResult(final, data, re_table, fe_table,
                                random_formula, selected_fe["formula"], fixed_name)
