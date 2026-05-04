# Tracking Upper Limb Use in Subacute Stroke: Investigating the Relationship Between Motor Functions and Real-World Arm Use
This sub-repository regroups the functions to analyze the association between arm activity performance (IMU-based metrics) and arm activity capacity (ARAT) across time post-stroke.

## Step 1: Setup your config file
In *.config\config_template.ini*

#### **Section: [Path_raw_data]**

Raw input files used before any preprocessing.

| Field	                | Description	                                      | Example         |
| :-------------------- | :-------------------------------------------------- | :-------------- | 
| home_path_to_data	    | Base directory containing clinical and IMU data     |	/database/data |
| folder_clinical_data	| Subfolder regrouping the clinical data	      |  Clinical_data/      |
| folder_redcap_file    | Subfolder containing the raw REDCap tables        | 20250930 |
| redcap_table_filename | Filename of the raw REDCap Excel sheet	          | raw_redcap.xlsx |
| folder_imu_data	    | Subfolder containing IMU-derived metrics	          | IMU_data/            |

All paths are constructed as:
home_path_to_data + folder_xxx + filename

#### **Section: [Path_additional_info]**

Metadata and auxiliary information used for preprocessing.


| Field	                               | Description	                                              | Required? | Example                             |
| :----------------------------------- | :----------------------------------------------------------- | :-------: | :---------------------------------- | 
| full_path_to_def_of_all_vars	       | Excel sheet mapping raw REDCap variable names to clean names |	Y         | metadata/variables_definitions.xlsx |
| full_path_to_database_protocol	   | File indicating expected assessment time points	          | Y         | metadata/protocol.xlsx              |
| full_path_to_redcap_data_corrections | CSV with manual corrections to recurring REDCap errors	      | N         | metadata/redcap_corrections.xlsx    |
| full_path_time_point_completion_imu  | Table describing missing or incomplete IMU measurements	  | Y         | metadata/Timepoint_completion_imu.xlsx |


#### **Section: [Path_processed_data]**

Paths where preprocessed files and generated figures will be saved.

| Field	                     | Description	                          | Example               |
| :------------------------- | :------------------------------------- | :-------------------- | 
| path_to_fig    	         | Output directory for figures           |	./fig/manuscript1/    |
| path_corrected_redcap_data | Name of the corrected REDCap XLSX file |  redcap_corrected.csv |
| path_formatted_redcap_data | REDCap table merged with IMU metrics   | redcap_IMU_merged.csv |


These files are created automatically by the pipeline.

#### **Section: [Variables_manuscript1]**

Variables controlling the mixed-effects model analysis.


| Field	            | Type      | Description	                                         | Example               |
| :---------------- | :-------: |:------------------------------------------------------ | :-------------------- | 
| var_of_interest   | str       | Metric to be modeled                                   | arm_use_ratio_VQF     |
| time_col          | str       | Variable describing time post-stroke                   | Days_post_stroke_tech |
| required_time_points | List[str] | Minimum list of required time points per participant for the analysis | 0, 5 |
| scale_factor_time | int       | Scaling factor to normalize time                       | 30                    | 
| grouping_var      | str       | Grouping column (e.g., patients ID)                    | ID                    |
| time_points_col   | str       | Column identifying discrete time points (T0, T3, etc.) | Tj                    |

**Bounds for filtering**

Each of these fields is a float, used to remove physiologically implausible values.

Examples include:

- upper_bound_arm_use_ratio_VQF
- lower_bound_relative_duration_use_VQF_impaired
- upper_bound_average_UL_activity_VQF_impaired

(and others as needed)

**Modeling parameters**

These specify the chosen fixed and random effects structures of the best-fitting MEM.

| Field	                   | Type      | Description	             | Example          |
| :----------------------- | :-------: |:--------------------------- | :--------------- |
| model_name               | str       | Model description           | quadratic_log_time | 
| fixed_effects_structure  | str       | Fixed effects in the model  | log1p_time_c     |
| random_effects_structure | str       | Random effects in the model | 1 + log1p_time_c |
| rounds_outcome           | bool      | Round outcome to int (for ARAT for example) | True   |

**Visualization**
| Field	                   | Type      | Description	             | Example          |
| :----------------------- | :-------: |:--------------------------- | :--------------- |
| color               | str       | Color for plotting           | tab:orange | 
| category  | str       | Categorization abbreviation  | P     |

**Potential predictors**

These specify which predictors could potentially best separate participants with significant change in capacity or performance. 
| Field	                   | Type      | Description	             | Example          |
| :----------------------- | :-------: |:--------------------------- | :--------------- |
| binary_predictors        | str       | List of categorical binary predictors Yes/No          | MEP_is_negative | 
| nominal_predictors       | str       | List of nominal predictors with 2 or more categories  | nat_str     |
| continuous_predictors    | str       | List of pseudo-continuous predictors  | fma_total    |
| target_k | int | Max number of predictors in model (depending the number of observations/IDs) | 3 |

**Basic usage of the scripts:**

    python scripts/manuscript1_perf_vs_capacity/main_scripts.py --config .config/config_manuscript1_template.ini \
    --var variable_of_interest 

where **main_scripts.py** must be adapted to the specific script, and variable_of_interest is optional and should specify the chosen outcome variable (e.g., **arm_use_ratio_VQF**).

# Dataset prepation: 
Prepare and format the latest redcap datasheet. Report typo errors in *Clinical_data/metadata/redcap_corrections.xlsx*. Run *scripts/clean_redcap_data/preprocessing_redcap_data_main.py* (need to adapt config file). Process IMU data with the *imu_processing_tailor_study* git.

Once processed, run *scripts/manuscript1_perf_vs_capacity/remove_invalid_trials.py* to copy Tj=1-2 to Tj=0 or 3, and remove the non-relevant time points. Run this script using the 3 modes: '', 'arat', and 'imu'.

## Analyze 1: Cross-sectional Correlation
Run *scripts/manuscript1_perf_vs_capacity/main_cross_sectional_correlations.py*. Don't forget to update *path_formatted_redcap_data* in *config_manuscript1_arm_use_tracking.ini* to the latest version of the corrected and cleaned REDCap data (i.e. output of *remove_invalid_trials.py*). 

## Analyze 2: Measuring Longitudinal Coupling
Run *scripts/manuscript1_perf_vs_capacity/main_longitudinal_coupling_perf_capacity.py* to model the coupling between performance and capacity over time. It first compute the delta scores between successive time points, then regress an OLS with participant-clustered robust SEs on the rate of change in performance with respect to the rate of change in capacity: 
$$
\text{rate}_{\mathrm{FAUR}} =
\beta_0
+ \beta_1\,\text{rate}_{\mathrm{ARAT}}
+ \beta_2\,\text{time}_{\mathrm{centered}}
+ \beta_3\,(\text{rate}_{\mathrm{ARAT}} \times \text{time}_{\mathrm{centered}})
+ \varepsilon
$$

Note: The piecewise 2-knots regression for [Lhose et al. 2025] is implemented in scripts/manuscript1_perf_vs_capacity/main_longitudinal_correlation.py*. It probably needs more than 3/4 observations per participant to converge nicely, but could be a nice approach for the analysis of the 3 years long folow-ups.

## Analyze 3: Find Best-Fitting Mixed Effect Model
Use *scripts/manuscript1_perf_vs_capacity/main_find_best_fitting_mem_model.py* to determine the optimal mixed-effects model for *var_of_interest* as a function of *time_col*. 

1. Select the optimal random-effects structure (REML mode). <br>
    Start with a fixed set of fixed effects and compare several candidate random-effects structures. <br>
    Choose the model with the lowest AIC, and retain its random-effects specification.

2. Optimize the fixed-effects structure (ML mode). <br>
    Using the selected random-effects structure, compare different fixed-effects formulations under maximum likelihood.<br>
    Identify the fixed-effects structure that minimizes AIC.

3. Refit the final model on the full dataset, run model diagnostics, & evaluate model's performance. <br>
    The final model is re-estimated using REML with the selected fixed and random effects. <br>
    Model diagnostics assess whether mixed-effects model assumptions are met. <br>
    Performance metrics are computed to understand how well the model generalizes to new patients.
    (Note: this evaluation is indicative, as the primary goal is not prediction but identifying the model structure that best represents the population in our sample.)


Use *scripts/manuscript1_perf_vs_capacity/main_show_plateau_times_in_one_fig.py* to plot the plateau times with associated CI for all performance metrics and capacity. 

Use *scripts/manuscript1_perf_vs_capacity/main_dual_plot_to_show_plateau.py* to plot capacity and performance average recovery curve on the same plot. 

## Analyze 4: Detect Participant with Significant Change
Use *scripts/manuscript1_perf_vs_capacity/main_detect_significant_change.py* to detect which participants significantly changed in capacity or performance with respect to baseline or model-expected change.

1. Fit the chosen MEM defined by config file (fixed and random structures). <br>
    The model (fixed + random effects structure defined in the config) is fit once. We extract:
    - population-level coefficients,
    - participant-specific random effects, 
    - covariance matrices and residual variance.

2. Simulate 1000 plausible trajectories per participant. <br>
    For each simulation (see [Lang et al., 2021](https://doi.org/10.1097/NPT.0000000000000413)):
    - sample new fixed-effect coefficients from their multivariate Normal distribution,
    - sample participant-specific random effects from their posterior distribution,
    - compute model-predicted values at last timepoint.

3. Compute simulated change scores. <br>
    Each simulation yields a predicted change between baseline and the chosen follow-up time. <br>
    This results in a distribution of 1000 plausible change scores per participant.

4. Compare observed vs. simulated change. <br>
    The participant’s actual/predicted change is compared against their simulation distribution to determine whether it is greater-than-zero or greater-than-expected.

Use *scripts/manuscript1_perf_vs_capacity/main_plot_change_detection.py* to visualize significant change detection in the population (Plots simulations' distributions per participant, pseudo p-values and z-scores, as well as CI).

Use *scripts/manuscript1_perf_vs_capacity/main_compare_all_performance_and_capacity_groups.py* to build contingency tables of P>0, P=0 vs C>0, C=0 and see how many participants have been detected per group. 

Use *scripts/manuscript1_perf_vs_capacity/main_plot_delta_vs_baseline.py* to identify which participants belong to which group, and check for ceiling effects.

Use *scripts/manuscript1_perf_vs_capacity/patients_demographics_per_subgroup.py* to create summary tables of the demographics, clinical, and performance metrics scores per subgroup: C+P+, CP, C+P, CP+.

Use *scripts/manuscript1_perf_vs_capacity/main_plot_population_avg_all.py* to plot all outcomes recovery average recovery curve with individuals' trajectories as well. 

## Analyze 5: Find Best Predictors for Participants with Significant Change
Use *scripts/manuscript1_perf_vs_capacity/main_find_best_predictors_of_categories.py* to perform a data-driven selection of the most important predictors from the hypothesis-based preselected set. 

1. Select candidate predictors: Remove highly correlated variables (e.g., ∣𝑟∣>0.7) to avoid redundancy.

2. Nested Cross-Validation for Selection:
    - Inner loop (feature selection): 
        Test all combinations of up to 3 predictors.
        Fit a penalized (ridge) logistic regression for classification for each combination of candidate predictors.
        Compute F1-score (balance of precision & recall).
        Track how often each predictor appears among top-performing subsets.

    - Outer loop (model validation):
        Use the most frequently selected predictors.
        Re-estimate the model and report its average F1-score.

3. Final Model on Full Dataset: Refit the ordered logistic regression with the most common predictors 


Use *scripts/manuscript1_perf_vs_capacity/main_violin_plots_to_visualize_predictor across_group.py* to visualize the distribution of the primary and secondary determinants of performance and capacity across "no improvement" and "significant improvement" groups. 

Use *scripts/manuscript1_perf_vs_capacity/main_plot_all_OR.py* to plot all forest plots. 