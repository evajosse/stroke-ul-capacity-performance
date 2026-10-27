"""Run the first-stage mixed-model selection pipeline on a CSV file."""
import argparse
import logging
from pathlib import Path

import pandas as pd
from model_selection import select_mixed_model
from individual_change import simulate_individual_change
from plotting import (population_trajectory, individual_change_forest,
                      example_change_distributions, report_covariance_diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/synthetic_longitudinal.csv')
    parser.add_argument('--outcome', default='arat')
    parser.add_argument('--time-col', default='days_post_stroke')
    parser.add_argument('--id-col', default='participant_id')
    parser.add_argument('--time-scale', type=float, default=30)
    parser.add_argument('--aic-tolerance', type=float, default=4)
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--n-parameter-draws', type=int, default=1000,
                        help='Gaussian fixed-effect draws for plateau/asymptote uncertainty')
    parser.add_argument('--plateau-fraction', type=float, default=0.90)
    parser.add_argument('--no-plots', action='store_true', help='Skip figure generation')
    parser.add_argument('--simulate-change', action='store_true', help='Run individual improvement simulations')
    parser.add_argument('--visit-col', default='timepoint')
    parser.add_argument('--baseline-visit', default='T0')
    parser.add_argument('--endpoint-visit', default='T4')
    parser.add_argument('--n-simulations', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0.84)
    parser.add_argument('--lower-bound', type=float, default=None)
    parser.add_argument('--upper-bound', type=float, default=None)
    parser.add_argument('--round-outcome', action='store_true')
    parser.add_argument('--direction', choices=['increase', 'decrease'], default='increase')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=out / 'fitting_diagnostics.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', force=True)
    print('Fitting candidate models (detailed diagnostics saved to fitting_diagnostics.log)...', flush=True)
    df = pd.read_csv(args.input)
    result = select_mixed_model(df, outcome=args.outcome, time_col=args.time_col,
                                group_col=args.id_col, scale_factor=args.time_scale,
                                aic_tolerance=args.aic_tolerance)
    result.random_comparison.to_csv(out / 'random_effects_comparison.csv', index=False)
    result.fixed_comparison.to_csv(out / 'fixed_effects_comparison.csv', index=False)
    summary = (f'Outcome: {args.outcome}\n'
               f'Selected random effects: {result.random_formula}\n'
               f'Selected fixed effects: {result.fixed_name} ({result.fixed_formula})\n\n'
               f'{result.model.summary()}\n')
    (out / 'selected_model_summary.txt').write_text(summary, encoding='utf-8')
    print(summary)
    diagnostics = report_covariance_diagnostics(result, out)
    if 'Warning:' in diagnostics[4]:
        print('Note: near-singular random-effects covariance; see covariance_diagnostics.txt')
    if not args.no_plots:
        population_trajectory(result, outcome=args.outcome, id_col=args.id_col,
                              time_col=args.time_col, scale_factor=args.time_scale,
                              output_dir=out, seed=args.seed,
                              n_parameter_draws=args.n_parameter_draws,
                              plateau_fraction=args.plateau_fraction)
    if args.simulate_change:
        default_bounds = {'arat': (0, 57), 'faur': (0, 2), 'faud': (0, 1), 'faui': (0, None)}
        low, high = default_bounds.get(args.outcome.lower(), (None, None))
        low = args.lower_bound if args.lower_bound is not None else low
        high = args.upper_bound if args.upper_bound is not None else high
        change = simulate_individual_change(
            result.model, result.data, outcome=args.outcome, id_col=args.id_col,
            visit_col=args.visit_col, baseline_visit=args.baseline_visit,
            endpoint_visit=args.endpoint_visit, n_simulations=args.n_simulations,
            improvement_probability=args.threshold, bounds=(low, high),
            round_to_int=args.round_outcome or args.outcome.lower() == 'arat',
            direction=args.direction, seed=args.seed)
        change.summary.to_csv(out / 'individual_change_summary.csv', index=False)
        change.changes.to_csv(out / 'simulated_change_distributions.csv', index=False)
        if not args.no_plots:
            individual_change_forest(change, id_col=args.id_col, outcome=args.outcome, output_dir=out)
            example_change_distributions(change, id_col=args.id_col, outcome=args.outcome, output_dir=out)
        print(f'Individual improvement: {int(change.summary.improved.sum())}/{len(change.summary)} participants; results saved.')
    print(f'Results saved to {out.resolve()}')


if __name__ == '__main__':
    main()
