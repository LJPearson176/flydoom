"""Tests for DOOM-001 Environment, Controllers, and Benchmark Protocol."""

import numpy as np
import pytest

from fly_doom.control.controllers import (
    BallisticForwardController,
    CompartmentalT4Controller,
    PointLIFMotionController,
    RandomController,
)
from fly_doom.doom.benchmark import Doom001Benchmark
from fly_doom.doom.interface import DoomAction, DoomObservation
from fly_doom.doom.mock_arena import MockDoomArena


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


def test_random_controller_benchmark_run():
    controller = RandomController(seed=42)
    benchmark = Doom001Benchmark(seeds=[1001, 1002], max_steps_per_episode=50)
    summary = benchmark.evaluate(controller)

    assert summary.num_episodes == 2
    assert summary.controller_name == "RandomController"
    assert len(summary.episodes) == 2
    for ep in summary.episodes:
        assert ep.survival_steps > 0
        assert ep.doom_score >= 0.0


def test_point_lif_vs_compartmental_controller():
    point_ctrl = PointLIFMotionController(width=32, height=32)
    comp_ctrl = CompartmentalT4Controller(width=32, height=32)

    env = MockDoomArena(width_px=32, height_px=32, max_steps=40)
    obs = env.reset(seed=42)

    act_point = point_ctrl.select_action(obs)
    act_comp = comp_ctrl.select_action(obs)

    assert act_point in list(DoomAction)
    assert act_comp in list(DoomAction)


def test_doom_benchmark_reproducibility():
    ctrl1 = RandomController(seed=999)
    ctrl2 = RandomController(seed=999)
    benchmark = Doom001Benchmark(seeds=[1001], max_steps_per_episode=30)

    res1 = benchmark.run_episode(ctrl1, seed=1001)
    res2 = benchmark.run_episode(ctrl2, seed=1001)

    assert res1.survival_steps == res2.survival_steps
    assert res1.doom_score == res2.doom_score
    assert res1.distance_traveled == res2.distance_traveled
