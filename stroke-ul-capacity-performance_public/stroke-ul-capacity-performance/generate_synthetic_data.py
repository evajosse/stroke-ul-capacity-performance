"""Generate illustrative longitudinal outcomes without using any patient records.

The ARAT-like outcome can have a quadratic or linear log-time population mean.
The other outcomes have linear log-time means. The fitted model is never forced.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_dataset(n_participants=40, seed=42, arat_trajectory="quadratic_log_time",
                     arat_curvature=-9.0, noise_sd=2.0):
    """Return five repeated visits per simulated participant.

    Time is log1p(months since the participant's first visit), consistent with
    model_selection.prepare_time_columns. Coefficients are illustrative only.
    """
    if n_participants < 3:
        raise ValueError("n_participants must be at least 3")
    if arat_trajectory not in {"quadratic_log_time", "linear_log_time"}:
        raise ValueError("Unknown ARAT trajectory")
    if noise_sd <= 0:
        raise ValueError("noise_sd must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    visits = [("T0", 21), ("T1", 42), ("T2", 60), ("T3", 90), ("T4", 180)]
    for pid in range(1, n_participants + 1):
        days = [nominal + int(rng.integers(-3, 4)) for _, nominal in visits]
        origin = min(days)
        intercept = rng.normal(16, 9)
        slope = rng.normal(25, 7.0) if arat_trajectory == "quadratic_log_time" else rng.normal(12, 3.0)
        # Separate individual effects produce heterogeneous IMU trajectories.
        imu_offsets = rng.normal(0, [0.10, 0.06, 1.4])
        imu_slopes = rng.normal(0, [0.055, 0.025, 0.7])
        for (visit, _), day in zip(visits, days):
            z = np.log1p((day - origin) / 30.0)
            curvature = arat_curvature * z**2 if arat_trajectory == "quadratic_log_time" else 0.0
            arat = np.clip(np.rint(intercept + slope*z + curvature + rng.normal(0, noise_sd)), 0, 57)
            # Independent, approximately linear-log-time population means.
            faur = np.clip(0.30 + 0.25*z + imu_offsets[0] + imu_slopes[0]*z + rng.normal(0, 0.06), 0, 2)
            faud = np.clip(0.09 + 0.09*z + imu_offsets[1] + imu_slopes[1]*z + rng.normal(0, 0.025), 0, 1)
            faui = max(0, 3.0 + 2.5*z + imu_offsets[2] + imu_slopes[2]*z + rng.normal(0, 0.8))
            rows.append(dict(participant_id=pid, timepoint=visit, days_post_stroke=day,
                             arat=int(arat), faur=round(float(faur), 4),
                             faud=round(float(faud), 4), faui=round(float(faui), 4)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/synthetic_longitudinal.csv')
    parser.add_argument('--n-participants', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arat-trajectory', choices=['quadratic_log_time', 'linear_log_time'],
                        default='quadratic_log_time')
    parser.add_argument('--arat-curvature', type=float, default=-9.0,
                        help='Coefficient of squared log-time; used only for quadratic example')
    parser.add_argument('--noise-sd', type=float, default=2.0)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(args.n_participants, args.seed, args.arat_trajectory,
                          args.arat_curvature, args.noise_sd)
    df.to_csv(path, index=False)
    print(f'Generated {len(df)} observations from {args.n_participants} synthetic participants: {path}')
    print(f'ARAT generating trajectory: {args.arat_trajectory} (curvature={args.arat_curvature if args.arat_trajectory == "quadratic_log_time" else 0})')


if __name__ == '__main__':
    main()
