"""Publication-oriented visualizations for the longitudinal mixed-model demo.

Population bands are Gaussian fixed-coefficient uncertainty intervals, not a
participant-level bootstrap. No individual random effects enter the population
curve. Plots are descriptive and do not change model selection or simulation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def population_trajectory(selection, *, outcome, id_col, time_col,
                          scale_factor=30, output_dir='results', seed=42,
                          n_parameter_draws=1000, plateau_fraction=0.90):
    """Observed participant trajectories and population fixed-effects curve.

    The curve uses the same globally centered time features as the model fit.
    Log-time is relative to each participant's first observed time; therefore
    the display grid is time since first visit, not necessarily time since stroke.
    """
    data, fit = selection.data, selection.model
    if not {'time_c', 'log1p_time_c', 'time_c2', 'log1p_time2_c'}.issubset(data):
        raise ValueError('Prepared time columns are required')
    origin = data.groupby(id_col)[time_col].transform('min')
    relative_months = (data[time_col] - origin) / scale_factor
    end = float(np.quantile(relative_months, .99))
    grid = np.linspace(0, max(end, 0.1), 240)
    # Global means implied by prepared columns; no need to recompute from raw data.
    t_center = float((data[time_col] / scale_factor - data['time_c']).mean())
    # Absolute-time linear term evaluated at a representative baseline (median).
    typical_baseline = float(np.median(origin)) / scale_factor
    linear = typical_baseline + grid - t_center
    log_mean = float((np.log1p(relative_months) - data['log1p_time_c']).mean())
    log = np.log1p(grid) - log_mean
    features = {'Intercept': np.ones_like(grid), 'time_c': linear,
                'time_c2': linear**2, 'log1p_time_c': log,
                'log1p_time2_c': log**2}
    names = list(fit.fe_params.index)
    X = np.column_stack([features[n] for n in names])
    beta = fit.fe_params.to_numpy(float)
    covariance = fit.cov_params().loc[names, names].to_numpy(float)
    mean = X @ beta
    se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', X, covariance, X), 0))
    lower, upper = mean - 1.96*se, mean + 1.96*se
    # Original plot_fixed_effects_LMM.py uses Gaussian fixed-effect parameter
    # draws for percentile uncertainty of the curve maximum/minimum and the
    # first 90%-of-improvement crossing. This is NOT a cluster bootstrap.
    # For the original implementation, the threshold of each draw is anchored
    # at the fitted population baseline (mean[0]), not the draw's own baseline.
    if n_parameter_draws < 1:
        raise ValueError('n_parameter_draws must be positive')
    if not 0 < plateau_fraction <= 1:
        raise ValueError('plateau_fraction must be in (0, 1]')
    rng = np.random.default_rng(seed)
    from individual_change import _draw_mvn
    beta_draws = _draw_mvn(rng, beta, covariance, size=n_parameter_draws)
    draws = beta_draws @ X.T
    # The original code chooses max for improving outcomes and min for
    # decreasing outcomes. Explicit direction avoids misclassifying scales.
    # This direction is based on the fitted curve's endpoint versus baseline.
    decreasing = bool(mean[-1] < mean[0])
    extremum = np.min if decreasing else np.max
    fitted_asymptote = float(extremum(mean))
    fitted_target = float(mean[0] + plateau_fraction * (fitted_asymptote - mean[0]))
    reached = mean <= fitted_target if decreasing else mean >= fitted_target
    plateau_month = float(grid[np.flatnonzero(reached)[0]]) if reached.any() else np.nan
    draw_asymptotes = extremum(draws, axis=1)
    draw_targets = mean[0] + plateau_fraction * (draw_asymptotes - mean[0])
    crossing = draws <= draw_targets[:, None] if decreasing else draws >= draw_targets[:, None]
    any_crossing = crossing.any(axis=1)
    draw_plateaus = np.full(n_parameter_draws, np.nan)
    draw_plateaus[any_crossing] = grid[np.argmax(crossing[any_crossing], axis=1)]
    asym_ci = np.quantile(draw_asymptotes, [.025, .975])
    plateau_ci = np.nanquantile(draw_plateaus, [.025, .975]) if any_crossing.any() else [np.nan, np.nan]
    summary = pd.DataFrame([{
        'outcome': outcome, 'plateau_fraction': plateau_fraction,
        'direction': 'decrease' if decreasing else 'increase',
        'asymptote_estimate': fitted_asymptote,
        'asymptote_ci_lower': asym_ci[0], 'asymptote_ci_upper': asym_ci[1],
        'plateau_months_since_first_visit': plateau_month,
        'plateau_ci_lower_months': plateau_ci[0],
        'plateau_ci_upper_months': plateau_ci[1],
        'n_parameter_draws': n_parameter_draws,
        'n_draws_with_crossing': int(any_crossing.sum()),
        'note': 'Parametric Gaussian fixed-effect draws; 90% of fitted improvement within displayed time grid; not a cluster bootstrap.'
    }])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out/'population_plateau_summary.csv', index=False)
    pd.DataFrame({'simulation': np.arange(1,n_parameter_draws+1),
                  'asymptote': draw_asymptotes,
                  'plateau_months_since_first_visit': draw_plateaus}).to_csv(
                      out/'population_plateau_draws.csv',index=False)
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    for _, grp in data.assign(_month=relative_months).groupby(id_col, sort=False):
        grp = grp.sort_values('_month')
        ax.plot(grp['_month'], grp[outcome], color='0.75', lw=.75, alpha=.35)
        ax.scatter(grp['_month'], grp[outcome], color='0.60', s=9, alpha=.3)
    ax.fill_between(grid, lower, upper, color='C0', alpha=.18, label='95% fixed-effect CI')
    ax.plot(grid, mean, color='C0', lw=2.5, label='Population fixed-effects fit')
    if np.isfinite(plateau_month):
        ax.axvline(plateau_month, color='C0', ls=':', lw=1.1, label=f'{plateau_fraction:.0%} improvement point')
        ax.axhline(fitted_target, color='C0', ls=':', lw=1.1)
    ax.set(xlabel='Months since first observed visit', ylabel=outcome,
           title=f'Population trajectory: {outcome}')
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, Path(output_dir)/'population_trajectory')
    pd.DataFrame({'months_since_first_visit':grid, 'estimate':mean,
                  'ci_lower':lower, 'ci_upper':upper}).to_csv(
                      Path(output_dir)/'population_trajectory_curve.csv', index=False)


def individual_change_forest(change, *, id_col, outcome, output_dir='results',
                             max_participants=80):
    """Median simulated changes, 95% simulation intervals and Q2 classification."""
    df = change.summary.copy().sort_values('simulated_change_median')
    if len(df) > max_participants:
        # A crowded forest plot is illegible; keep the most extreme and middle values.
        inds = np.unique(np.linspace(0, len(df)-1, max_participants).astype(int))
        df = df.iloc[inds]
    y = np.arange(len(df))
    med = df['simulated_change_median'].to_numpy(float)
    lo = df['simulated_change_p025'].to_numpy(float)
    hi = df['simulated_change_p975'].to_numpy(float)
    improved = df['improved'].to_numpy(bool)
    fig, ax = plt.subplots(figsize=(9, max(5, .24*len(df)+1.5)))
    for state, label, color in [(False, 'Not classified as improved', '0.45'),
                                (True, 'Classified as improved', 'C0')]:
        mask = improved == state
        if mask.any():
            ax.errorbar(med[mask], y[mask],
                        xerr=np.vstack([med[mask]-lo[mask], hi[mask]-med[mask]]),
                        fmt='o', ms=3.7, capsize=2, lw=.8, color=color,
                        label=label)
    ax.axvline(0, ls='--', color='0.25', lw=1)
    ax.set(yticks=y, yticklabels=df[id_col].astype(str),
           xlabel=f'Simulated change in {outcome} (endpoint − baseline)',
           ylabel='Participant', title='Individual simulated change (95% interval)')
    ax.legend(frameon=False, loc='best')
    fig.tight_layout()
    _save(fig, Path(output_dir)/'individual_change_forest')


def example_change_distributions(change, *, id_col, outcome,
                                 output_dir='results', n_examples=3):
    """Deterministically select participants spanning improvement probabilities."""
    summary = change.summary.sort_values('probability_improvement')
    if summary.empty:
        raise ValueError('No participant simulation results')
    idx = np.unique(np.linspace(0, len(summary)-1,
                                min(n_examples, len(summary))).astype(int))
    examples = summary.iloc[idx]
    fig, axes = plt.subplots(len(examples), 1, figsize=(8, 2.7*len(examples)),
                             squeeze=False)
    for ax, (_, participant) in zip(axes[:,0], examples.iterrows()):
        pid = participant[id_col]
        vals = change.changes.loc[change.changes[id_col] == pid, 'change'].to_numpy(float)
        ax.hist(vals, bins=32, color='C0', alpha=.75, edgecolor='white', linewidth=.3)
        ax.axvline(0, color='0.2', ls='--', lw=1.3)
        ax.set(title=f'{id_col}={pid} | P(improvement)={participant["probability_improvement"]:.3f}',
               xlabel=f'Simulated change in {outcome}', ylabel='Frequency')
    fig.tight_layout()
    _save(fig, Path(output_dir)/'example_change_distributions')


def report_covariance_diagnostics(selection, output_dir='results'):
    """Record, rather than conceal, possible near-singular random effects."""
    G = np.asarray(selection.model.cov_re, dtype=float)
    eig = np.linalg.eigvalsh((G + G.T)/2)
    condition = float(np.linalg.cond(G))
    lines = ['Random-effects covariance diagnostics',
             f'Eigenvalues: {eig.tolist()}',
             f'Condition number: {condition:.4g}',
             f'Residual variance: {selection.model.scale:.6g}',
             'Warning: covariance is near singular; interpret random-effect uncertainty cautiously.'
             if eig[0] <= 1e-6*max(eig[-1], 1) or condition > 1e6
             else 'No near-singularity detected by the simple eigenvalue/condition-number check.',
             'This is a numerical diagnostic, not a formal model adequacy test.']
    Path(output_dir, 'covariance_diagnostics.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return lines
