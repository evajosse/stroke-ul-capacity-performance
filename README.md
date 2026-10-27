# Longitudinal upper limb recovery: mixed-effects models and individual change detection

This repository provides a **standalone, standardized implementation of the longitudinal mixed-effects modeling and model-based individual improvement detection** used in:

> Josse E, Cheng H-J, Chin LF, et al. **Wearable-Based Digital Metrics Reveal Dynamic Dissociations Between Upper Limb Capacity and Real-World Use Over the First Six Months After Stroke.** Research Square preprint (2026). [https://doi.org/10.21203/rs.3.rs-9813697/v1](https://doi.org/10.21203/rs.3.rs-9813697/v1).

**Publication status:** The link above points to the preprint. The final journal article DOI will be added when available.

## Scientific rationale

After stroke, recovery is often evaluated through standardized clinical tests of **capacity**—what a person can do under assessment conditions. Actual **performance**, such as use of the paretic arm in everyday life, may evolve differently. The accompanying study followed upper limb capacity (Action Research Arm Test, ARAT) and wearable-derived functional arm use over the first six months after stroke to examine these differences.

Identifying *individual* improvement is difficult when measurements are noisy and recovery trajectories differ across people. Fixed thresholds such as the **Minimal Detectable Change (MDC)** or **Minimal Clinically Important Difference (MCID)** may vary with recovery stage, population and outcome, and are not always appropriate for comparing different measurement constructs. The paper therefore used a **model-based, uncertainty-aware approach**: longitudinal mixed-effects models characterize recovery, and posterior predictive simulations estimate each participant's probability of improving between baseline and follow-up. This is a statistical classification of positive change, **not** a claim that the magnitude of change is clinically important.

This repository makes that portion of the analysis usable on a tidy longitudinal CSV without the original study's REDCap infrastructure or private patient data.

## Scope

**Included:**

- Selection of random-effects and fixed-effects time structures for longitudinal mixed-effects models.
- Population-level recovery curves with pointwise Gaussian fixed-effects confidence intervals.
- Model-based estimation of a 90%-of-fitted-improvement time, with percentile intervals from Gaussian fixed-effect parameter draws.
- Conditional best linear unbiased prediction (**BLUP**) of individual random effects using pre-endpoint observations.
- Simulation of individual baseline-to-follow-up change, incorporating fixed-effects uncertainty, conditional random-effects uncertainty and residual variation, with optional outcome bounds and integer rounding.
- Classification of improvement using the estimated probability of positive change (default **≥ 0.84**), plus plots and CSV outputs.

**Not included:** the paper's cross-sectional correlations, longitudinal capacity–performance coupling regression, baseline predictor modeling, and other study-specific analyses. This code does not reconstruct the original clinical cohort or reproduce its numerical findings without the underlying data.

## Installation

Python **3.11** is recommended. In a terminal, from the repository folder:

```bash
conda create -n stroke-recovery python=3.11
conda activate stroke-recovery
python -m pip install -r requirements.txt
```

If you already have an appropriate Conda environment, activate it and install the requirements without creating another one.

## Quick start: reproducible synthetic example

Generate an **illustrative** dataset with 40 participants, five visits per participant, and ARAT, FAUR, FAUD and FAUI outcomes:

```bash
python generate_synthetic_data.py --n-participants 40 --seed 42
```

This creates `data/synthetic_longitudinal.csv`. The default synthetic ARAT generating trajectory is quadratic in log time; the three simulated functional arm-use outcomes have linear log-time generating trajectories. These choices demonstrate model selection and **are not estimates from the published cohort**.

Fit the ARAT model, run 1,000 individual change simulations, and generate plots:

```bash
python run_analysis.py --input data/synthetic_longitudinal.csv --outcome arat --simulate-change --n-simulations 1000 --n-parameter-draws 1000
```

Outputs are written to `results/`. For a different synthetic outcome, change `--outcome` to `faur`, `faud` or `faui`.

Run the automated tests:

```bash
python -m pytest -q
```

## Use with your own data

Supply a CSV in **long format**, with one row per participant per visit. By default the following columns are expected:

| Column | Meaning | Example |
| --- | --- | --- |
| `participant_id` | Participant identifier | `101` |
| `timepoint` | Visit label | `T0`, `T1`, `T2`, `T3`, `T4` |
| `days_post_stroke` | Numeric time since stroke, in days | `21`, `90`, `180` |
| `arat` | Outcome to model (or another numeric outcome column) | `24` |

Participant/visit pairs must be unique. For individual-change simulations, participants must have the specified baseline and endpoint visits and at least three observations. The code uses only observations **before the endpoint** to condition participant-specific random effects; the population model itself is fitted to all included observations. This is not independent prospective validation.

Example with your own visit labels and an outcome named `score`:

```bash
python run_analysis.py --input data/my_longitudinal_data.csv --outcome score --id-col participant_id --time-col days_post_stroke --visit-col timepoint --baseline-visit T1 --endpoint-visit T4 --lower-bound 0 --upper-bound 57 --round-outcome --simulate-change --n-simulations 1000
```

The code does **not** automatically apply the original study's protocol-specific exclusions or missingness criteria. Prepare and quality-check your dataset before analysis. For outcomes with different scales, explicitly set the appropriate bounds and direction. Default bounds are ARAT `[0,57]`, FAUR `[0,2]`, FAUD `[0,1]` (fraction), and FAUI `[0,∞)`; for a differently named outcome, no bounds are assumed.

For a decreasing outcome (lower is better), use `--direction decrease`. The default is `increase`. Run `python run_analysis.py --help` for all options. **The `--direction` flag governs individual change classification; population plateau calculations infer direction from the fitted trajectory.**

### Time definitions

The model uses time in units of `days_post_stroke / --time-scale` (default divisor: 30). Candidate fixed effects are linear time, log time, quadratic time and quadratic log time. Linear time is globally mean-centered. Log time is referenced to each participant's first observed visit, transformed with `log1p`, then globally mean-centered. Accordingly, the population plots and plateau times are expressed in **months since first observed visit**, not directly in months since stroke. Do not interpret them as absolute post-stroke plateau times without accounting for the baseline visit timing.

## Statistical methods

**Model selection.** Candidate random-effects structures are evaluated first using a linear fixed time effect. Fixed-effects time formulations are subsequently compared using maximum likelihood. The simplest model within **4 AIC points** of the minimum is retained. The selected model is refitted using restricted maximum likelihood (REML). The random-effects selection stage uses a manually computed REML-AIC criterion consistent with the original analysis implementation; users should be aware that REML likelihood comparisons across different fixed-effects structures are generally inappropriate, which is why the fixed-effects stage uses ML.

**Individual improvement.** For each eligible participant, the fitted model supplies a conditional Gaussian distribution of individual random effects given observations before the endpoint. Repeated draws from the estimated fixed-effects Gaussian distribution, the conditional random-effects distribution, and a residual distribution generate simulated baseline and endpoint outcomes. Residual draws are truncated when outcome bounds are specified. The fraction of simulations with strictly positive change (or strictly negative change for a decreasing outcome) is the estimated improvement probability. The default classification threshold is **probability ≥ 0.84**. The threshold is a methodological choice from the research analysis, not an MDC/MCID threshold.

**Population curves and plateau.** Pointwise confidence bands use the fitted fixed-effect covariance and a Gaussian approximation. Uncertainty in the estimated 90%-of-improvement time and curve extremum is estimated by simulating Gaussian fixed-effect parameter draws and reporting percentiles. The reported curve extremum is within the **displayed observation horizon**, not necessarily a mathematical asymptote. The 90% threshold is anchored to the fitted baseline value. These calculations are *parametric simulations*, not participant-level bootstrap resampling.

## Outputs

All outputs are written under `results/` (or a directory specified with `--output-dir`).

| File | Contents |
| --- | --- |
| `random_effects_comparison.csv` | Random-effects candidate comparisons |
| `fixed_effects_comparison.csv` | Fixed-effects candidate comparisons |
| `selected_model_summary.txt` | Selected model and fitted coefficients |
| `fitting_diagnostics.log` | Detailed fitting diagnostics |
| `covariance_diagnostics.txt` | Random-effects covariance diagnostics |
| `population_trajectory_curve.csv` | Population curve and pointwise intervals |
| `population_plateau_summary.csv` | Estimated 90%-of-improvement point and interval |
| `population_plateau_draws.csv` | Fixed-effect simulation results for plateau estimation |
| `population_trajectory.png` / `.pdf` | Population recovery trajectory |
| `individual_change_summary.csv` | Per-participant improvement probability and classification |
| `simulated_change_distributions.csv` | Simulated change values for each participant |
| `individual_change_forest.png` / `.pdf` | Per-participant simulated changes and intervals |
| `example_change_distributions.png` / `.pdf` | Example participant change distributions |

The last four output types are generated only when `--simulate-change` is supplied; figures can be suppressed with `--no-plots`. Population plateau CSVs are generated as part of the plotting step.

## Citation

If this implementation supports your work, please cite the associated paper:

Josse E, et al. *Wearable-Based Digital Metrics Reveal Dynamic Dissociations Between Upper Limb Capacity and Real-World Use Over the First Six Months After Stroke.* Research Square preprint (2026). https://doi.org/10.21203/rs.3.rs-9813697/v1

This reference will be updated when the final journal DOI becomes available.

## AI-assisted code standardization disclosure

**Large language models (LLMs) were used to assist with refactoring, documenting, testing and standardizing the author's original research code into a standalone pipeline that is not tailored to the original dataset.** The statistical methodology originated from the research analysis described in the associated paper. The author reviewed the implementation and conducted numerical and synthetic-data validation checks. AI assistance does not imply that the original study data or results were generated by an LLM.
