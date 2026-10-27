"""Smoke-check selection under two distinct synthetic data-generating means."""
import logging
from generate_synthetic_data import generate_dataset
from model_selection import select_mixed_model


def test_recovers_default_quadratic_and_linear_examples(tmp_path):
    logging.getLogger("recovery.model_selection").addHandler(logging.NullHandler())
    for trajectory, expected in (("quadratic_log_time", "quadratic_log_time"),
                                 ("linear_log_time", "log_time")):
        data = generate_dataset(40, seed=42, arat_trajectory=trajectory)
        result = select_mixed_model(data, "arat")
        assert result.fixed_name == expected
        assert result.model.converged
