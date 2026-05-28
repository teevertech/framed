"""Custom SB3 callbacks for the panel-sequencing training loop.

``AimTrainingCallback``
    Logs per-episode training metrics (reward, collision rate, travel
    time, penalty cost) to aim on every episode end across all workers.
    Also forwards PPO internal stats (policy loss, value loss, entropy,
    KL, clip fraction) from SB3's logger at the start of each rollout.

``EvalCallback``
    Runs on a fixed schedule (``eval_freq`` timesteps).  For each of a
    set of fixed evaluation panels it records:
      - greedy-nearest episode reward
      - greedy-cost-aware episode reward
      - current policy episode reward (deterministic)
    Also saves a side-by-side GIF (greedy nearest vs current policy) to
    ``{gif_dir}/{run_name}/step_{N:08d}.gif`` at each eval cycle so you
    can watch the policy improve over training.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from framed.baselines import greedy_cost_aware_action, greedy_nearest_action, run_episode
from framed.config import logger
from framed.env import PanelEnv
from framed.panel import Panel

if TYPE_CHECKING:
    from aim import Run
    from framed.config import TrainConfig


class AimTrainingCallback(BaseCallback):
    """Per-episode training metrics → aim.

    Accumulates per-step stats (collision flag, travel time) for each
    parallel worker and flushes to aim when that worker's episode ends.
    Also forwards SB3's internal PPO stats (policy/value/entropy loss,
    approx KL, clip fraction) at the start of each rollout.
    """

    def __init__(self, aim_run: Run, n_envs: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.aim_run = aim_run
        self._n_envs = n_envs
        self._ep_reward:    list[float] = [0.0] * n_envs
        self._ep_travel:    list[float] = [0.0] * n_envs
        self._ep_collisions: list[int]  = [0]   * n_envs

    def _on_rollout_start(self) -> None:
        stats = getattr(self.model.logger, "name_to_value", {})
        for name, value in stats.items():
            try:
                self.aim_run.track(float(value), name=f"ppo/{name}",
                                   step=self.num_timesteps)
            except (TypeError, ValueError):
                pass

    def _on_step(self) -> bool:
        for i, (info, reward, done) in enumerate(zip(
            self.locals["infos"],
            self.locals["rewards"],
            self.locals["dones"],
        )):
            self._ep_reward[i]     += float(reward)
            self._ep_travel[i]     += float(info.get("travel_time", 0.0))
            self._ep_collisions[i] += int(info.get("collided", False))
            if done:
                self._flush(i, info)
        return True

    def _flush(self, w: int, info: dict) -> None:
        step    = self.num_timesteps
        n_steps = max(int(info.get("step_count", 1)), 1)
        r, t    = self._ep_reward[w], self._ep_travel[w]
        self.aim_run.track(r,             name="train/ep_reward",             step=step)
        self.aim_run.track(t,             name="train/total_travel_time",     step=step)
        self.aim_run.track(r + t,         name="train/collision_penalty_cost", step=step)
        self.aim_run.track(self._ep_collisions[w] / n_steps,
                           name="train/collision_rate", step=step)
        self._ep_reward[w] = self._ep_travel[w] = 0.0
        self._ep_collisions[w] = 0


class EvalCallback(BaseCallback):
    """Periodic evaluation on fixed panels → aim + GIF snapshots.

    Every ``eval_freq`` timesteps, runs three policies on each eval panel
    and logs mean episode rewards to aim.  If ``gif_dir`` is set, also
    saves a side-by-side GIF (greedy nearest vs current policy) on the
    first eval panel — one GIF per eval cycle, named by timestep so they
    sort chronologically and you can watch the policy improve over time.

    GIF path: ``{gif_dir}/{run_name}/step_{N:08d}.gif``

    Parameters
    ----------
    config:
        ``TrainConfig`` for env construction params.
    eval_panels:
        Fixed panels; reused every eval cycle for fair comparison.
    aim_run:
        Active ``aim.Run`` instance.
    run_name:
        Label used to namespace GIF output paths.
    eval_freq:
        Evaluation interval in environment timesteps.
    gif_dir:
        Root directory for GIFs.  Set to ``''`` to disable.
    gif_fps:
        Frame rate for saved GIFs.
    verbose:
        SB3 verbosity level.
    """

    def __init__(
        self,
        config: TrainConfig,
        eval_panels: list[Panel],
        aim_run: Run,
        run_name: str = "run",
        eval_freq: int = 10_000,
        gif_dir: Path | None = None,
        gif_fps: int = 20,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._config          = config
        self._aim_run         = aim_run
        self._eval_freq       = eval_freq
        self._last_eval_step  = 0
        self._gif_fps         = gif_fps
        self._gif_dir         = gif_dir

        if self._gif_dir is not None:
            self._gif_dir.mkdir(parents=True, exist_ok=True)

        self._eval_envs: list[PanelEnv] = [
            PanelEnv(panel,
                     robot_speed=config.robot_speed,
                     collision_penalty_multiplier=config.collision_penalty_multiplier)
            for panel in eval_panels
        ]

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self._eval_freq:
            self._run_eval()
            self._last_eval_step = self.num_timesteps
        return True

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def _run_eval(self) -> None:
        n_scores, c_scores, p_scores = [], [], []
        for env in self._eval_envs:
            n_scores.append(run_episode(env, greedy_nearest_action)[0])
            c_scores.append(run_episode(env, greedy_cost_aware_action)[0])
            p_scores.append(self._eval_policy(env))

        step = self.num_timesteps
        self._aim_run.track(float(np.mean(n_scores)), name="eval/greedy_nearest",    step=step)
        self._aim_run.track(float(np.mean(c_scores)), name="eval/greedy_cost_aware", step=step)
        self._aim_run.track(float(np.mean(p_scores)), name="eval/policy_reward",     step=step)

        if self.verbose >= 1:
            logger.info(
                f"[eval @ {step:,}]  "
                f"policy={np.mean(p_scores):.2f}  "
                f"nearest={np.mean(n_scores):.2f}  "
                f"cost_aware={np.mean(c_scores):.2f}"
            )

        if self._gif_dir is not None:
            self._save_gif(step)

    def _eval_policy(self, env: PanelEnv) -> float:
        obs, _ = env.reset()
        total  = 0.0
        for _ in range(env.n_members):
            action, _ = self.model.predict(
                obs, action_masks=env.action_masks(), deterministic=True
            )
            obs, reward, terminated, _, _ = env.step(int(action))
            total += float(reward)
            if terminated:
                break
        return total

    # ------------------------------------------------------------------ #
    # GIF snapshot                                                         #
    # ------------------------------------------------------------------ #

    def _save_gif(self, step: int) -> None:
        """Side-by-side GIF: greedy nearest (left) vs current policy (right)."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            from matplotlib.patches import Polygon

            from framed.visualize import (
                _PATH_CLEAR, _PATH_COLLIDE, _ROBOT_COLOR,
                _collect_frames, _draw_legend, _draw_member,
                _figure_size, _robot_size, _robot_triangle, _setup_axes,
            )

            env   = self._eval_envs[0]
            panel = env.panel
            model = self.model

            def _model_policy(e: PanelEnv) -> int:
                action, _ = model.predict(
                    e.obs, action_masks=e.action_masks(), deterministic=True
                )
                return int(action)

            fpm, pf  = 10, 3
            frames_n = _collect_frames(env, greedy_nearest_action, fpm, pf)
            frames_p = _collect_frames(env, _model_policy,         fpm, pf)

            n = max(len(frames_n), len(frames_p))
            while len(frames_n) < n: frames_n.append(frames_n[-1])
            while len(frames_p) < n: frames_p.append(frames_p[-1])

            fw, fh = _figure_size(panel)
            fig, (ax_l, ax_r) = plt.subplots(
                1, 2, figsize=(fw * 2 + 0.3, fh),
                gridspec_kw={"wspace": 0.32},
            )
            fig.patch.set_facecolor("#FAFAFA")
            rsize = _robot_size(panel)

            def _render(ax, frame, title: str) -> None:
                ax.cla()
                _setup_axes(ax, panel)
                ax.set_title(f"{title}  ·  {frame.total_reward:.1f}",
                             fontsize=8, pad=3, color="#333")
                for m in panel.members:
                    _draw_member(ax, m,
                                 placed=m.id in frame.placed_ids,
                                 is_target=m.id == frame.target_id)
                for fp, tp, col in frame.paths:
                    c = _PATH_COLLIDE if col else _PATH_CLEAR
                    ax.annotate("", xy=tp, xytext=fp,
                                arrowprops=dict(arrowstyle="-|>", color=c,
                                                lw=1.0, mutation_scale=7),
                                zorder=3)
                if frame.partial_path:
                    fp, tp = frame.partial_path
                    c = _PATH_COLLIDE if frame.collided_this else _PATH_CLEAR
                    ax.plot([fp[0], tp[0]], [fp[1], tp[1]],
                            color=c, lw=1.0, alpha=0.7, zorder=3)
                tri = _robot_triangle(frame.robot_xy, frame.direction, rsize)
                ax.add_patch(Polygon(tri, closed=True, facecolor=_ROBOT_COLOR,
                                     edgecolor="white", lw=0.7, zorder=5))

            def _draw(i: int) -> None:
                _render(ax_l, frames_n[i], "Greedy Nearest")
                _render(ax_r, frames_p[i], f"Policy @ {step:,} steps")
                _draw_legend(ax_r, panel)

            anim = FuncAnimation(fig, _draw, frames=n,
                                 interval=1000 / self._gif_fps,
                                 repeat=False, blit=False)
            gif_path = self._gif_dir / f"step_{step:08d}.gif"
            anim.save(str(gif_path), writer="pillow", fps=self._gif_fps, dpi=100)
            plt.close(fig)

            if self.verbose >= 1:
                logger.info(f"gif saved → {gif_path}")

        except Exception as exc:
            if self.verbose >= 1:
                logger.warning(f"could not save gif at step {step}: {exc}")
