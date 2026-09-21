import numpy as np
import pandas as pd
import pytest

from src.extra.dataset_experiment.canonical_adapter import (
    _apply_input_mapping,
    _load_cache,
    _measured_pair_days,
    _record,
    _regular_profile,
    _resample_pair_days,
    _save_cache,
)
from src.extra.ieee33_device_day_simulation.population.device_day_loader import DeviceDayPool


def _config():
    return {
        "population": {"canonical_adapter": {
            "fallback_capacity_kwh": 10.0,
            "fallback_peak_power_kw": 1.0,
            "fallback_soc_fraction": 0.5,
        }},
        "zones": {"zones": {
            f"zone_{index}": {"buses": [index + 1]} for index in range(1, 7)
        }},
    }


def test_regular_profile_preserves_contract_and_scale():
    profile = _regular_profile(np.arange(24, dtype=float), scale=2.0)
    assert profile.shape == (288,)
    assert np.isfinite(profile).all()
    assert profile.min() >= 0.0
    assert profile.max() <= 46.0


def test_pair_resampling_keeps_external_input_nonzero():
    index = pd.date_range("2024-01-01", periods=48, freq="30min")
    result = _resample_pair_days(
        pd.Series(np.ones(48), index=index),
        pd.Series(np.linspace(0.0, 2.0, 48), index=index),
        max_days=1,
    )
    assert len(result) == 1
    _, load, external, timestamp = result[0]
    assert load.shape == external.shape == timestamp.shape == (288,)
    assert external.max() > 1.9


def test_record_rejects_negative_external_input():
    with pytest.raises(ValueError, match="invalid 288-point"):
        _record(
            np.ones(288), "source", 0, 0, _config(),
            timestamp=np.arange(288), source_unit="kw", energy_input=-np.ones(288),
        )


def test_cache_round_trip(tmp_path):
    record = _record(
        np.ones(288), "source", 0, 0, _config(),
        timestamp=np.arange(288), source_unit="kw", energy_input=np.full(288, 0.5),
    )
    path = tmp_path / "canonical.npz"
    _save_cache(path, DeviceDayPool([record], 1, 1, 0))
    pool = _load_cache(path)
    assert pool is not None
    assert pool.source_count == 1
    assert pool.records[0].load_kw.shape == (288,)
    assert np.allclose(pool.records[0].energy_input_kw, 0.5)


def test_counterfactual_input_crosses_real_load():
    profile = np.linspace(0.2, 4.0, 288)
    record = _record(
        profile, "source", 0, 0, _config(),
        timestamp=np.arange(288), source_unit="kw",
    )
    pool = _apply_input_mapping(
        DeviceDayPool([record], 1, 1, 0),
        {"input_mapping": "load_shape_counterfactual", "counterfactual_input_scale": 1.0},
    )
    balance = pool.records[0].energy_input_kw - profile
    assert np.any(balance > 0)
    assert np.any(balance < 0)


def test_unique_source_batches_do_not_duplicate_within_snapshot():
    records = []
    for source_index in range(12):
        for day in range(2):
            records.append(_record(
                np.ones(288), f"source-{source_index}", day, source_index, _config(),
                timestamp=np.arange(288), source_unit="kw",
            ))
    pool = DeviceDayPool(records, len(records), 12, 0)
    batches = pool.sample_source_unique_batches(per_zone=2, batch_count=2, seed=7)
    assert len(batches) == 2
    assert all(len({record.source_device_id for record in batch}) == 12 for batch in batches)
    assert {record.source_device_id for record in batches[0]} == {
        record.source_device_id for record in batches[1]
    }


def test_measured_only_pair_keeps_missing_external_input_zero():
    index = pd.date_range("2024-01-01", periods=48, freq="30min")
    load = pd.Series(np.linspace(0.2, 2.0, len(index)), index=index)
    profiles = _measured_pair_days(load, pd.Series(dtype=float), max_days=1)

    assert len(profiles) == 1
    _, mapped_load, mapped_input, _ = profiles[0]
    assert mapped_load.max() > mapped_load.min()
    assert np.all(mapped_input == 0.0)


def test_counterfactual_mapping_can_be_forbidden_per_dataset():
    record = _record(
        np.ones(288), "source", 0, 0, _config(),
        timestamp=np.arange(288), source_unit="kw",
    )
    with pytest.raises(ValueError, match="forbids counterfactual"):
        _apply_input_mapping(
            DeviceDayPool([record], 1, 1, 0),
            {
                "input_mapping": "load_shape_counterfactual",
                "forbid_counterfactual_input": True,
            },
        )


def test_record_accepts_audited_baseline_and_device_parameters():
    baseline = np.linspace(0.0, 2.0, 288)
    record = _record(
        np.ones(288), "source", 0, 0, _config(),
        timestamp=np.arange(288), source_unit="kw", baseline=baseline,
        capacity_kwh=12.0, peak_power_kw=4.0, soc_fraction=0.6,
    )

    assert np.allclose(record.baseline_battery_kw, baseline)
    assert record.capacity_kwh == 12.0
    assert record.peak_power_kw == 4.0
    assert record.initial_soc == pytest.approx(0.6)
