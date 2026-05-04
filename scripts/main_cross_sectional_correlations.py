"""
Generate a 3x4 subplot of VARS_PERFORMANCE metrics versus VAR_CAPACITY metrics at T1, T2, T3, and T4. On each scatter plot, a linear OLS is fitted between the 2 cosntructs and the Spearman correlation with boostraping CI is indicated to measure cross-sectional correlations up to 6 months post-stroke. 
"""
import os
import argparse
from pathlib import Path
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
from pandas import Series

from utilities.import_files import load_config
from utilities.exclude_patients_or_time_points_from_analysis import exclude_patients_without_required_time_points
from manuscipt1_perf_vs_capacity.correlations.correlations_per_Tj_grid import plot_cross_sectional_grid_capacity_vs_performance

VAR_CAPACITY = 'arat_tot'
VARS_PERFORMANCE = ("arm_use_ratio_VQF", "relative_duration_use_VQF_impaired", "average_intensity_active_VQF_impaired")

def run_pipeline(
    config_path: str | Path, 
) -> None:
    config_file_path = os.path.join(r".config", config_path)
    config_file_path = Path(config_file_path)
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    print(f"[INFO] Using config: {config_path}")
    cfg = load_config(
        config_path,
        clinical_data_formatted=True,
        vars_def=True
    )

    # ----- Extract data and parameters -----
    df: pd.DataFrame = cfg["redcap_data_formatted"] 
    # Use: formatted_data_v2025-12-18_IMU_metrics_with_optimized_thres_cleaned_for_imu.csv
    vars_def = cfg["vars_def"]


    fig_dir = os.path.join(cfg["figures"], "cross_sectional_correlation")
    os.makedirs(fig_dir, exist_ok=True)

    # ----- Clean dataset -----
    # df = exclude_patients_without_required_time_points(
    #     df,
    #     required_time_points=required_Tj,
    #     outcome_col=outcome_var,
    # )
    # df = df.dropna(subset=[outcome_var, time_col])


    print(f"n_obs = {len(df)}, n_subjects = {df['ID'].nunique()}")

    # ----- STEP 1 -----

    plot_cross_sectional_grid_capacity_vs_performance(
        df=df,
        varDef=vars_def,
        time_points=(0, 3, 4, 5),
        capacity_var="arat_tot",

        # Put YOUR three performance variables here:
        metrics_ratio=("arm_use_ratio_VQF",),  # or your FAUR/ratio metric
        metrics_duration=("relative_duration_use_VQF_impaired",),
        metrics_intensity=("average_intensity_active_VQF_impaired",),

        out_dir=fig_dir,
        fig_name="grid_ARAT_vs_ratio_duration_intensity_T1_T4_presentation.png",
        title="Cross-sectional association between capacity (ARAT) and  performance metrics (Ratio, Duration, & Intensity)",
        n_boot=1000,   # 10k is fine too; 5k often enough for a plot
        seed=0,
    )

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the command-line interface.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the full mixed-effects model pipeline for manuscript 1: "
            "best random-effects, best fixed-effects, final fit + evaluation."
        )
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config_manuscript1_arm_use_tracking.ini", #config_manuscript1_template.ini",
        help=(
            "Path to the INI configuration file "
            "(default: config_manuscript1_template.ini)"
        ),
    )
    parser.add_argument(
        "-v", "--var",
        type=str,
        default=None,
        help=(
            "Optional: override `var_of_interest` defined in the config file "
            "(e.g. arm_use_ratio_VQF)."
        ),
    )
    return parser


if __name__ == "__main__":
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()
    run_pipeline(config_path=args.config)
