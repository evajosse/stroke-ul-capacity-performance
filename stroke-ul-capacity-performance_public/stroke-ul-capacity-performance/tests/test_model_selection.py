import numpy as np
import pandas as pd
from model_selection import prepare_time_columns, _choose, select_mixed_model


def test_time_preparation():
    df = pd.DataFrame({"participant_id": [1, 1, 2, 2],
                       "days_post_stroke": [20, 50, 25, 55]})
    result = prepare_time_columns(df, "days_post_stroke", "participant_id")
    assert np.isclose(result["log1p_time_c"].mean(), 0)
    assert np.isclose(result["time_c"].mean(), 0)
    assert np.isclose(result.loc[0, "log1p_time_c"], result.loc[2, "log1p_time_c"])


def test_four_point_parsimony():
    table = pd.DataFrame({"name": ["linear", "quadratic"],
                          "aic": [103.9, 100.0], "converged": [True, True]})
    assert _choose(table, ["linear", "quadratic"], 4)["name"] == "linear"
    assert _choose(table, ["linear", "quadratic"], 3)["name"] == "quadratic"


def test_full_model_smoke():
    rng = np.random.default_rng(42)
    rows = []
    for pid in range(24):
        intercept = rng.normal(15, 4)
        slope = rng.normal(4, 1)
        for t in (20, 45, 90, 180):
            y = intercept + slope * np.log1p((t - 20) / 30) + rng.normal(0, 2)
            rows.append((pid, t, y))
    df = pd.DataFrame(rows, columns=["participant_id", "days_post_stroke", "outcome"])
    result = select_mixed_model(df, "outcome")
    assert result.model.converged
    assert len(result.random_comparison) == 3
    assert len(result.fixed_comparison) == 4
