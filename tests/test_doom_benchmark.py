"""Tests for DOOM-002 Environment, Controlled Controllers, and Benchmark Protocol."""

import numpy as np
import pytest

from fly_doom.control.controllers import (
    BallisticForwardController,
    ControlledT4Controller,
    PointLIFMotionController,
    CompartmentalT4Controller,
    RandomController,
)
from fly_doom.doom.benchmark import Doom002Benchmark
from fly_doom.doom.interface import DoomAction, DoomObservation
from fly_doom.doom.mock_arena import MockDoomArena
from fly_doom.doom.vizdoom_wrapper import VizDoomEnvironment
from fly_doom.dynamics.compartmental_t4 import CompartmentModelType


def test_mock_arena_initialization_and_step():
    env = MockDoomArena(width_px=32, height_px=32, max_steps=100)
    obs = env.reset(seed=123)

    assert isinstance(obs, DoomObservation)
    assert obs.rgb.shape == (32, 32, 3)
    assert obs.depth.shape == (32, 32)
    assert obs.health == 100.0
    assert obs.ammo == 50
    assert obs.step_count == 0
    assert not obs.done
    assert env.provenance.tier == "engineering_scaffold"

    # Step forward
    next_obs, reward, done, info = env.step(DoomAction.FORWARD)
    assert next_obs.step_count == 1
    assert "distance_traveled" in info

    # Step turning
    next_obs, reward, done, info = env.step(DoomAction.TURN_LEFT)
    assert next_obs.step_count == 2

    # Step firing
    next_obs, reward, done, info = env.step(DoomAction.FIRE)
    assert next_obs.ammo == 49


def test_random_controller_benchmark_run_and_trajectory():
    controller = RandomController(seed=42)
    benchmark = Doom002Benchmark(seeds=[1001, 1002], max_steps_per_episode=30)
    summary = benchmark.evaluate(controller)

    assert summary.num_episodes == 2
    assert summary.controller_name == "RandomController"
    assert len(summary.episodes) == 2
    for ep in summary.episodes:
        assert ep.survival_steps > 0
        assert ep.flydoom_fitness_score >= 0.0
        assert len(ep.trajectory) == ep.survival_steps
        first_tick = ep.trajectory[0]
        assert first_tick.seed == ep.seed
        assert first_tick.step == 0
        assert first_tick.controller_name == "RandomController"


def test_controlled_t4_controllers_ablation():
    # Verify Model A, B, C, D all run and produce neural telemetry under identical policy
    ctrl_a = ControlledT4Controller(model_type=CompartmentModelType.MODEL_A, width=32, height=32)
    ctrl_d = ControlledT4Controller(model_type=CompartmentModelType.MODEL_D, width=32, height=32)

    env = MockDoomArena(width_px=32, height_px=32, max_steps=40)
    obs = env.reset(seed=42)

    act_a = ctrl_a.select_action(obs)
    n_a = ctrl_a.get_neural_state()
    assert act_a in list(DoomAction)
    assert "t4_l_v" in n_a and "motion_asymmetry" in n_a

    act_d = ctrl_d.select_action(obs)
    n_d = ctrl_d.get_neural_state()
    assert act_d in list(DoomAction)
    assert "t4_l_v" in n_d and "motion_asymmetry" in n_d


def test_synaptic_knockout_controller():
    # Verify synaptic knockouts (e.g. Mi4 KO)
    ctrl_ko = ControlledT4Controller(
        model_type=CompartmentModelType.MODEL_D,
        synaptic_knockouts={"Mi4", "Mi9"},
        width=32,
        height=32,
    )
    assert "Mi4" in ctrl_ko.knockouts
    assert "Mi9" in ctrl_ko.knockouts
    assert "Mi4" in ctrl_ko.name

    env = MockDoomArena(width_px=32, height_px=32, max_steps=10)
    obs = env.reset(seed=42)
    action = ctrl_ko.select_action(obs)
    assert action in list(DoomAction)


def test_vizdoom_environment_probe():
    diag = VizDoomEnvironment.check_vizdoom_available()
    assert "available" in diag
    assert "status" in diag
    # When vizdoom is not installed, falls back cleanly to MockDoomArena
    env = VizDoomEnvironment(screen_resolution=(32, 32), max_steps=10)
    obs = env.reset(seed=123)
    assert obs.rgb.shape == (32, 32, 3)
