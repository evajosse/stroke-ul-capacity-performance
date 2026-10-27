import numpy as np
from generate_synthetic_data import generate_dataset
from model_selection import select_mixed_model
from individual_change import simulate_individual_change


def test_simulation_reproducible_and_bounded():
    df = generate_dataset(n_participants=12, seed=42)
    selected = select_mixed_model(df, outcome='arat')
    kw = dict(outcome='arat', n_simulations=30, bounds=(0, 57), round_to_int=True, seed=9)
    a = simulate_individual_change(selected.model, selected.data, **kw)
    b = simulate_individual_change(selected.model, selected.data, **kw)
    assert len(a.summary) == 12
    assert len(a.changes) == 12 * 30
    assert np.array_equal(a.changes.change.to_numpy(), b.changes.change.to_numpy())
    assert a.changes.change.between(-57, 57).all()
    assert a.summary.probability_improvement.between(0, 1).all()
