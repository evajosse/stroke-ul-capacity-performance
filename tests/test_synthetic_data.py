from generate_synthetic_data import generate_dataset


def test_synthetic_data_reproducible_and_bounded():
    a = generate_dataset(8, seed=5)
    b = generate_dataset(8, seed=5)
    assert a.equals(b)
    assert len(a) == 40
    assert a.groupby('participant_id').size().eq(5).all()
    assert a.arat.between(0, 57).all()
    assert a.faur.between(0, 2).all()
    assert a.faud.between(0, 1).all()
    assert a.faui.ge(0).all()


def test_trajectory_modes_differ_and_are_bounded():
    quadratic = generate_dataset(12, seed=42, arat_trajectory="quadratic_log_time")
    linear = generate_dataset(12, seed=42, arat_trajectory="linear_log_time")
    assert not quadratic["arat"].equals(linear["arat"])
    assert linear["arat"].between(0, 57).all()
    assert quadratic["arat"].between(0, 57).all()
