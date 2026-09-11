"""Tests for multi-compartment T4 dendritic simulation engine."""

import pytest
from fly_doom.connectome.t4_anatomical import build_canonical_t4a_reconstruction
from fly_doom.dynamics.compartmental_t4 import (
    CompartmentalParameters,
    CompartmentalT4Engine,
    CompartmentModelType,
)


def test_compartmental_t4_quiescent_state():
    recon = build_canonical_t4a_reconstruction()
    params = CompartmentalParameters()

    for m_type in [
        CompartmentModelType.MODEL_A,
        CompartmentModelType.MODEL_B,
        CompartmentModelType.MODEL_C,
        CompartmentModelType.MODEL_D,
    ]:
        engine = CompartmentalT4Engine(recon, params, model_type=m_type)
        # Advance 100 steps with zero input
        for _ in range(100):
            fired = engine.step(0.0, 0.0, 0.0, 0.0)
            assert not fired
        assert abs(engine.v_soma - params.v_rest) < 1e-4


def test_compartmental_t4_spike_generation():
    recon = build_canonical_t4a_reconstruction()
    params = CompartmentalParameters()

    for m_type in [
        CompartmentModelType.MODEL_A,
        CompartmentModelType.MODEL_B,
        CompartmentModelType.MODEL_C,
        CompartmentModelType.MODEL_D,
    ]:
        engine = CompartmentalT4Engine(recon, params, model_type=m_type)
        spikes = 0
        for _ in range(100):
            if engine.step(in_mi1=8.0, in_tm3=8.0, in_mi4=0.0, in_mi9=0.0):
                spikes += 1
        assert spikes > 0, f"Engine under {m_type} failed to fire spikes with strong input"


def test_model_d_shunting_inhibition():
    recon = build_canonical_t4a_reconstruction()
    params = CompartmentalParameters()

    # Engine with excitation only
    engine_exc = CompartmentalT4Engine(recon, params, model_type=CompartmentModelType.MODEL_D)
    spikes_exc = 0
    for _ in range(100):
        if engine_exc.step(in_mi1=5.0, in_tm3=5.0, in_mi4=0.0, in_mi9=0.0):
            spikes_exc += 1

    # Engine with identical excitation plus leading shunting inhibition
    engine_shunt = CompartmentalT4Engine(recon, params, model_type=CompartmentModelType.MODEL_D)
    spikes_shunt = 0
    for _ in range(100):
        if engine_shunt.step(in_mi1=5.0, in_tm3=5.0, in_mi4=10.0, in_mi9=5.0):
            spikes_shunt += 1

    assert spikes_shunt < spikes_exc, f"Shunting failed to suppress firing: {spikes_shunt} vs {spikes_exc}"
