"""DOOM-003 Disentangled Task-Competence Benchmark Protocol & Telemetry Recorder.

Evaluates embodied agents across deterministic seeds, recording 4 independent dimensions:
  1. NAVIGATION: goal_distance_delta, min_dist_to_goal, nav_progress_pct, exit_reached, wall_bumps
  2. COMBAT: kills, damage_dealt, shots_fired, hit_accuracy (damage / (shots * 25))
  3. SURVIVAL: survival_steps, health_remaining, damage_taken
  4. NEURAL DYNAMICS: dsi, mean_asymmetry, std_asymmetry, spike_rate

Plus secondary composite index:
  - FlyDoom Fitness Score: 100 * kills + 1.0 * damage_dealt + 0.5 * distance + 20.0 * nav_progress + 0.1 * survival_steps

Full Per-Tick Neural Telemetry (trajectory.parquet):
  step, seed, controller, x, y, angle, health, ammo, action, reward,
  retina_mean, retina_delta, t4_l_v, t4_r_v, t4_l_spike, t4_r_spike,
  motion_asymmetry, norm_asymmetry, center_depth, dist_to_goal
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from fly_doom.core.provenance import Provenance
from fly_doom.doom.interface import DoomAction, DoomEnvironment, DoomObservation
from fly_doom.doom.mock_arena import MockDoomArena


@dataclass
class TrajectoryTick:
    """Full causal trace logged at every simulation frame."""

    seed: int
    controller_name: str
    step: int
    x: float
    y: float
    angle_rad: float
    health: float
    ammo: int
    action: int
    action_name: str
    reward: float
    retina_mean: float
    retina_delta: float
    t4_l_v: float
    t4_r_v: float
    t4_l_spike: float
    t4_r_spike: float
    motion_asymmetry: float
    norm_asymmetry: float
    center_depth: float
    dist_to_goal: float


@dataclass
class EpisodeResult:
    """Comprehensive multi-dimensional fitness metrics for a single evaluation episode."""

    seed: int
    controller_name: str

    # 1. NAVIGATION METRICS
    nav_progress_pct: float
    min_dist_to_goal: float
    final_dist_to_goal: float
    exit_reached: bool
    distance_traveled: float
    wall_bumps: int

    # 2. COMBAT METRICS
    kills: int
    damage_dealt: float
    damage_taken: float
    shots_fired: int
    hit_accuracy: float

    # 3. SURVIVAL METRICS
    survival_steps: int
    health_remaining: float
    ammo_remaining: int
    total_reward: float

    # 4. NEURAL METRICS
    mean_norm_asymmetry: float
    std_norm_asymmetry: float
    spike_rate_left: float
    spike_rate_right: float

    # Composite index
    flydoom_fitness_score: float
    action_counts: Dict[str, int]
    trajectory: List[TrajectoryTick]


@dataclass
class BenchmarkSummary:
    """Aggregated multi-episode statistics (Mean ± SEM) across all dimensions."""

    controller_name: str
    num_episodes: int

    # Composite
    fitness_score_mean: float
    fitness_score_sem: float

    # Navigation
    nav_progress_mean: float
    nav_progress_sem: float
    distance_traveled_mean: float
    distance_traveled_sem: float
    min_dist_mean: float
    min_dist_sem: float
    exit_rate: float
    wall_bumps_mean: float

    # Combat
    kills_mean: float
    kills_sem: float
    damage_dealt_mean: float
    damage_dealt_sem: float
    hit_accuracy_mean: float
    hit_accuracy_sem: float

    # Survival
    survival_steps_mean: float
    survival_steps_sem: float
    health_mean: float

    # Neural
    std_norm_asymmetry_mean: float
    spike_rate_mean: float

    episodes: List[EpisodeResult]


class Doom003Benchmark:
    """Standardized DOOM-003 Task-Competence Benchmark Protocol."""

    DEFAULT_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        max_steps_per_episode: int = 250,
        env_factory: Optional[Callable[[], DoomEnvironment]] = None,
    ):
        self.seeds = seeds or self.DEFAULT_SEEDS
        self.max_steps = max_steps_per_episode
        self.env_factory = env_factory or (lambda: MockDoomArena(max_steps=self.max_steps))

    def run_episode(
        self,
        controller: Any,
        seed: int,
        env: Optional[DoomEnvironment] = None,
    ) -> EpisodeResult:
        should_close = False
        if env is None:
            env = self.env_factory()
            should_close = True

        if hasattr(controller, "reset"):
            controller.reset()

        obs = env.reset(seed=seed)
        total_reward = 0.0
        action_counts = {act.name: 0 for act in DoomAction}
        trajectory: List[TrajectoryTick] = []
        cname = getattr(controller, "name", controller.__class__.__name__)

        while not obs.done and obs.step_count < self.max_steps:
            action = controller.select_action(obs)
            action_int = int(action)
            act_name = DoomAction(action_int).name
            action_counts[act_name] += 1

            n_state: Dict[str, float] = {}
            if hasattr(controller, "get_neural_state"):
                n_state = controller.get_neural_state()

            next_obs, reward, done, info = env.step(action)
            total_reward += reward

            tick = TrajectoryTick(
                seed=seed,
                controller_name=cname,
                step=obs.step_count,
                x=float(obs.x),
                y=float(obs.y),
                angle_rad=float(obs.angle_rad),
                health=float(obs.health),
                ammo=int(obs.ammo),
                action=action_int,
                action_name=act_name,
                reward=float(reward),
                retina_mean=n_state.get("retina_mean", 0.0),
                retina_delta=n_state.get("retina_delta", 0.0),
                t4_l_v=n_state.get("t4_l_v", -65.0),
                t4_r_v=n_state.get("t4_r_v", -65.0),
                t4_l_spike=n_state.get("t4_l_spike", 0.0),
                t4_r_spike=n_state.get("t4_r_spike", 0.0),
                motion_asymmetry=n_state.get("motion_asymmetry", 0.0),
                norm_asymmetry=n_state.get("norm_asymmetry", 0.0),
                center_depth=n_state.get("center_depth", float(np.mean(obs.depth[:, 26:38])) if obs.depth is not None else 10.0),
                dist_to_goal=info.get("dist_to_goal", 14.5),
            )
            trajectory.append(tick)
            obs = next_obs

        # Compute Disentangled Metrics
        shots = max(1, info.get("shots_fired", 0))
        accuracy = min(1.0, info.get("damage_dealt", 0.0) / (shots * 25.0))
        nav_progress = float(info.get("nav_progress", 0.0))

        # Neural summary metrics from trajectory
        asyms = [t.norm_asymmetry for t in trajectory] if trajectory else [0.0]
        spikes_l = sum(t.t4_l_spike for t in trajectory) / max(1, len(trajectory))
        spikes_r = sum(t.t4_r_spike for t in trajectory) / max(1, len(trajectory))

        # Composite FlyDoom Fitness Score
        fitness_score = (
            100.0 * obs.kill_count
            + 1.0 * obs.damage_dealt
            + 0.5 * info.get("distance_traveled", 0.0)
            + 50.0 * max(0.0, nav_progress)
            + (100.0 if info.get("exit_reached", False) else 0.0)
            + 0.1 * obs.step_count
        )

        res = EpisodeResult(
            seed=seed,
            controller_name=cname,
            nav_progress_pct=round(nav_progress * 100.0, 2),
            min_dist_to_goal=round(info.get("min_dist_to_goal", 14.5), 2),
            final_dist_to_goal=round(info.get("dist_to_goal", 14.5), 2),
            exit_reached=bool(info.get("exit_reached", False)),
            distance_traveled=round(info.get("distance_traveled", 0.0), 3),
            wall_bumps=int(info.get("wall_bumps", 0)),
            kills=obs.kill_count,
            damage_dealt=obs.damage_dealt,
            damage_taken=info.get("damage_taken", 0.0),
            shots_fired=shots,
            hit_accuracy=round(accuracy, 3),
            survival_steps=obs.step_count,
            health_remaining=obs.health,
            ammo_remaining=obs.ammo,
            total_reward=round(total_reward, 2),
            mean_norm_asymmetry=round(float(np.mean(asyms)), 4),
            std_norm_asymmetry=round(float(np.std(asyms)), 4),
            spike_rate_left=round(float(spikes_l), 4),
            spike_rate_right=round(float(spikes_r), 4),
            flydoom_fitness_score=round(fitness_score, 2),
            action_counts=action_counts,
            trajectory=trajectory,
        )

        if should_close:
            env.close()

        return res

    def evaluate(self, controller: Any) -> BenchmarkSummary:
        env = self.env_factory()
        episodes: List[EpisodeResult] = []

        for seed in self.seeds:
            ep_res = self.run_episode(controller, seed=seed, env=env)
            episodes.append(ep_res)

        env.close()

        n = len(episodes)
        cname = getattr(controller, "name", controller.__class__.__name__)

        def _mean_sem(arr: List[float]) -> Tuple[float, float]:
            m = float(np.mean(arr))
            sem = float(np.std(arr, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            return round(m, 3), round(sem, 3)

        fit_m, fit_sem = _mean_sem([e.flydoom_fitness_score for e in episodes])
        nav_m, nav_sem = _mean_sem([e.nav_progress_pct for e in episodes])
        dist_m, dist_sem = _mean_sem([e.distance_traveled for e in episodes])
        min_d_m, min_d_sem = _mean_sem([e.min_dist_to_goal for e in episodes])
        exit_r = float(sum(1 for e in episodes if e.exit_reached)) / n
        bumps_m = float(np.mean([e.wall_bumps for e in episodes]))

        k_m, k_sem = _mean_sem([e.kills for e in episodes])
        d_m, d_sem = _mean_sem([e.damage_dealt for e in episodes])
        acc_m, acc_sem = _mean_sem([e.hit_accuracy for e in episodes])

        surv_m, surv_sem = _mean_sem([e.survival_steps for e in episodes])
        hp_m = float(np.mean([e.health_remaining for e in episodes]))

        std_asym_m = float(np.mean([e.std_norm_asymmetry for e in episodes]))
        spk_m = float(np.mean([e.spike_rate_left + e.spike_rate_right for e in episodes]))

        return BenchmarkSummary(
            controller_name=cname,
            num_episodes=n,
            fitness_score_mean=fit_m,
            fitness_score_sem=fit_sem,
            nav_progress_mean=nav_m,
            nav_progress_sem=nav_sem,
            distance_traveled_mean=dist_m,
            distance_traveled_sem=dist_sem,
            min_dist_mean=min_d_m,
            min_dist_sem=min_d_sem,
            exit_rate=exit_r,
            wall_bumps_mean=round(bumps_m, 1),
            kills_mean=k_m,
            kills_sem=k_sem,
            damage_dealt_mean=d_m,
            damage_dealt_sem=d_sem,
            hit_accuracy_mean=acc_m,
            hit_accuracy_sem=acc_sem,
            survival_steps_mean=surv_m,
            survival_steps_sem=surv_sem,
            health_mean=round(hp_m, 1),
            std_norm_asymmetry_mean=round(std_asym_m, 4),
            spike_rate_mean=round(spk_m, 4),
            episodes=episodes,
        )


# Backward-compatible aliases
Doom001Benchmark = Doom003Benchmark
Doom002Benchmark = Doom003Benchmark
