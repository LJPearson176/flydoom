"""DOOM-001 Standard Sensorimotor Benchmark Protocol.

Evaluates an embodied agent across 10 deterministic seeds, recording:
  - Doom Score: 100 * kills + 1.0 * damage_dealt + 0.5 * navigation_progress + 0.1 * survival_time
  - Survival time (steps lived)
  - Kills (imps eliminated)
  - Damage dealt
  - Distance traveled
  - Action entropy / distribution
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from fly_doom.core.provenance import Provenance
from fly_doom.doom.interface import DoomAction, DoomEnvironment, DoomObservation
from fly_doom.doom.mock_arena import MockDoomArena


@dataclass
class EpisodeResult:
    """Telemetry and fitness metrics for a single evaluation episode."""

    seed: int
    controller_name: str
    survival_steps: int
    kills: int
    damage_dealt: float
    distance_traveled: float
    health_remaining: float
    ammo_remaining: int
    total_reward: float
    doom_score: float
    action_counts: Dict[str, int]


@dataclass
class BenchmarkSummary:
    """Aggregated multi-episode statistics (Mean ± SEM) for a controller."""

    controller_name: str
    num_episodes: int
    doom_score_mean: float
    doom_score_sem: float
    survival_steps_mean: float
    survival_steps_sem: float
    kills_mean: float
    kills_sem: float
    damage_dealt_mean: float
    damage_dealt_sem: float
    distance_traveled_mean: float
    distance_traveled_sem: float
    episodes: List[EpisodeResult]


class Doom001Benchmark:
    """Standardized DOOM-001 Benchmark Protocol."""

    DEFAULT_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        max_steps_per_episode: int = 1000,
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
        """Run a single benchmark episode."""
        should_close = False
        if env is None:
            env = self.env_factory()
            should_close = True

        if hasattr(controller, "reset"):
            controller.reset()

        obs = env.reset(seed=seed)
        total_reward = 0.0
        action_counts = {act.name: 0 for act in DoomAction}

        while not obs.done and obs.step_count < self.max_steps:
            action = controller.select_action(obs)
            action_counts[DoomAction(int(action)).name] += 1

            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            obs = next_obs

        # Calculate composite DOOM-001 score
        # Weighting: 100 per kill + 1.0 per damage + 0.5 per distance + 0.1 per step survived
        doom_score = (
            100.0 * obs.kill_count
            + 1.0 * obs.damage_dealt
            + 0.5 * info.get("distance_traveled", 0.0)
            + 0.1 * obs.step_count
        )

        res = EpisodeResult(
            seed=seed,
            controller_name=getattr(controller, "name", controller.__class__.__name__),
            survival_steps=obs.step_count,
            kills=obs.kill_count,
            damage_dealt=obs.damage_dealt,
            distance_traveled=info.get("distance_traveled", 0.0),
            health_remaining=obs.health,
            ammo_remaining=obs.ammo,
            total_reward=total_reward,
            doom_score=doom_score,
            action_counts=action_counts,
        )

        if should_close:
            env.close()

        return res

    def evaluate(self, controller: Any) -> BenchmarkSummary:
        """Evaluate a controller across all benchmark seeds and compute summary statistics."""
        env = self.env_factory()
        episodes: List[EpisodeResult] = []

        for seed in self.seeds:
            ep_res = self.run_episode(controller, seed=seed, env=env)
            episodes.append(ep_res)

        env.close()

        n = len(episodes)
        cname = getattr(controller, "name", controller.__class__.__name__)

        scores = [e.doom_score for e in episodes]
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
            doom_score_mean=s_m,
            doom_score_sem=s_sem,
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
