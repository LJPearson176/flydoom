"""DOOM-002 Standard Sensorimotor Benchmark Protocol & Telemetry Recorder.

Evaluates embodied agents across deterministic seeds, recording:
  - FlyDoom Fitness Score: 100 * kills + 1.0 * damage_dealt + 0.5 * distance + 0.1 * survival_steps
  - Raw Metrics: Kills, Damage Dealt, Distance Traveled, Survival Steps, Health, Ammo, Native Reward
  - Full Per-Tick Neural Telemetry (trajectory.parquet):
      step, seed, controller, x, y, angle, health, ammo, action, reward,
      retina_mean, retina_delta, t4_l_v, t4_r_v, t4_l_spike, t4_r_spike,
      motion_asymmetry, center_depth
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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
    center_depth: float


@dataclass
class EpisodeResult:
    """Summary fitness metrics for a single evaluation episode."""

    seed: int
    controller_name: str
    survival_steps: int
    kills: int
    damage_dealt: float
    distance_traveled: float
    health_remaining: float
    ammo_remaining: int
    total_reward: float
    flydoom_fitness_score: float
    action_counts: Dict[str, int]
    trajectory: List[TrajectoryTick]


@dataclass
class BenchmarkSummary:
    """Aggregated multi-episode statistics (Mean ± SEM) for a controller."""

    controller_name: str
    num_episodes: int
    fitness_score_mean: float
    fitness_score_sem: float
    survival_steps_mean: float
    survival_steps_sem: float
    kills_mean: float
    kills_sem: float
    damage_dealt_mean: float
    damage_dealt_sem: float
    distance_traveled_mean: float
    distance_traveled_sem: float
    episodes: List[EpisodeResult]


class Doom002Benchmark:
    """Standardized DOOM-002-MOCK Benchmark Protocol."""

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
        """Run a single benchmark episode logging full tick-by-tick neural telemetry."""
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

            # Query controller neural state if available
            n_state: Dict[str, float] = {}
            if hasattr(controller, "get_neural_state"):
                n_state = controller.get_neural_state()

            next_obs, reward, done, info = env.step(action)
            total_reward += reward

            # Log frame tick
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
                center_depth=n_state.get("center_depth", float(np.mean(obs.depth[:, 26:38])) if obs.depth is not None else 10.0),
            )
            trajectory.append(tick)
            obs = next_obs

        # Composite FlyDoom Fitness Score
        fitness_score = (
            100.0 * obs.kill_count
            + 1.0 * obs.damage_dealt
            + 0.5 * info.get("distance_traveled", 0.0)
            + 0.1 * obs.step_count
        )

        res = EpisodeResult(
            seed=seed,
            controller_name=cname,
            survival_steps=obs.step_count,
            kills=obs.kill_count,
            damage_dealt=obs.damage_dealt,
            distance_traveled=info.get("distance_traveled", 0.0),
            health_remaining=obs.health,
            ammo_remaining=obs.ammo,
            total_reward=total_reward,
            flydoom_fitness_score=fitness_score,
            action_counts=action_counts,
            trajectory=trajectory,
        )

        if should_close:
            env.close()

        return res

    def evaluate(self, controller: Any) -> BenchmarkSummary:
        """Evaluate a controller across all benchmark seeds."""
        env = self.env_factory()
        episodes: List[EpisodeResult] = []

        for seed in self.seeds:
            ep_res = self.run_episode(controller, seed=seed, env=env)
            episodes.append(ep_res)

        env.close()

        n = len(episodes)
        cname = getattr(controller, "name", controller.__class__.__name__)

        scores = [e.flydoom_fitness_score for e in episodes]
        survivals = [e.survival_steps for e in episodes]
        kills = [e.kills for e in episodes]
        damages = [e.damage_dealt for e in episodes]
        dists = [e.distance_traveled for e in episodes]

        def _mean_sem(arr: List[float]) -> Tuple[float, float]:
            m = float(np.mean(arr))
            sem = float(np.std(arr, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            return round(m, 3), round(sem, 3)

        s_m, s_sem = _mean_sem(scores)
        surv_m, surv_sem = _mean_sem(survivals)
        k_m, k_sem = _mean_sem(kills)
        d_m, d_sem = _mean_sem(damages)
        dist_m, dist_sem = _mean_sem(dists)

        return BenchmarkSummary(
            controller_name=cname,
            num_episodes=n,
            fitness_score_mean=s_m,
            fitness_score_sem=s_sem,
            survival_steps_mean=surv_m,
            survival_steps_sem=surv_sem,
            kills_mean=k_m,
            kills_sem=k_sem,
            damage_dealt_mean=d_m,
            damage_dealt_sem=d_sem,
            distance_traveled_mean=dist_m,
            distance_traveled_sem=dist_sem,
            episodes=episodes,
        )


# Backward-compatible alias
Doom001Benchmark = Doom002Benchmark
