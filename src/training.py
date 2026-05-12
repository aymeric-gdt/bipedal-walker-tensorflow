"""
Boucle d'entraînement principale pour PPO sur BipedalWalker-v3.

Inclut :
  - Multi-env parallel (SubprocVecEnv ou DummyVecEnv)
  - Logging TensorBoard
  - Checkpointing
  - Early stopping
  - Multi-seed support
"""

import os
import time
import numpy as np
from datetime import datetime
from collections import deque

import tensorflow as tf
try:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False

from src.config import PPOConfig
from src.agent import PPOAgent


def setup_gpu(config: PPOConfig):
    """Configure GPU pour TensorFlow : memory growth + mixed precision."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if config.gpu_memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs detected: {len(gpus)}")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        except RuntimeError as e:
            print(f"GPU config error: {e}")
    else:
        print("WARNING: No GPU detected — training will run on CPU")

    if config.use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision (float16) enabled")
    else:
        print("Mixed precision disabled (float32)")
    print()


def make_vec_env(env_id: str, n_envs: int = 8, seed: int = 0,
                 use_subproc: bool = True):
    """Crée un vectorized environment.

    Args:
        env_id: Nom de l'environnement Gymnasium
        n_envs: Nombre d'envs parallèles
        seed: Graine pour le premier env (les suivants seed + 1, +2, ...)
        use_subproc: Utiliser SubprocVecEnv (plus rapide) ou DummyVecEnv
    """
    if not _HAS_SB3:
        raise ImportError("stable_baselines3 requis pour make_vec_env: pip install stable-baselines3")

    def make_env(rank: int):
        def _init():
            import gymnasium as gym
            env = gym.make(env_id)
            env.reset(seed=seed + rank)
            return env
        return _init

    env_fns = [make_env(i) for i in range(n_envs)]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


class TensorBoardLogger:
    """Logger TensorBoard natif TensorFlow — pas de dépendance torch."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
        self.writer.flush()

    def log_scalars(self, tag_group: str, values: dict, step: int):
        with self.writer.as_default():
            for k, v in values.items():
                tf.summary.scalar(f"{tag_group}/{k}", v, step=step)
        self.writer.flush()

    def close(self):
        self.writer.flush()
        self.writer.close()


class TrainingRunner:
    """Boucle d'entraînement PPO complète."""

    def __init__(self, config: PPOConfig, env_id: str = "BipedalWalker-v3",
                 log_dir: str = None, seed: int = 0):
        self.config = config
        self.env_id = env_id
        self.seed = seed

        # Log dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ppo_{env_id}_s{seed}_{timestamp}"
        self.log_dir = log_dir or f"runs/{run_name}"
        os.makedirs(self.log_dir, exist_ok=True)

        # TensorBoard
        self.tb = TensorBoardLogger(self.log_dir)

        # Env
        self.env = make_vec_env(
            env_id,
            n_envs=config.n_envs,
            seed=seed,
            use_subproc=(config.n_envs > 1),
        )

        # Agent
        # Get obs_dim et action_dim depuis l'env
        dummy_obs = self.env.reset()
        self.obs_dim = dummy_obs.shape[1]
        self.action_dim = self.env.action_space.shape[0]

        self.agent = PPOAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            config=config,
        )

        # État
        self.global_step = 0
        self.iteration = 0
        self.wall_start = time.time()

        # Reward history pour early stopping
        self.reward_history = deque(maxlen=config.early_stop_window)

        print(f"=== Entraînement PPO — {env_id} ===")
        print(f"Seed: {seed}")
        print(f"Log: {self.log_dir}")
        print(f"TensorBoard: tensorboard --logdir={self.log_dir}")
        print(f"n_envs: {config.n_envs}, n_steps: {config.n_steps}")
        print(f"Total timesteps: {config.total_timesteps:,}")
        print(f"Observations: {self.obs_dim}D, Actions: {self.action_dim}D")

    def _log(self, rollout_stats: dict, train_stats: dict):
        """Log vers TensorBoard — métriques riches PPO."""
        step = self.global_step

        # --- Rollout ---
        self.tb.log_scalar("rollout/ep_rew_mean", rollout_stats.get("mean_ep_reward", 0.0), step)
        self.tb.log_scalar("rollout/ep_len_mean", rollout_stats.get("mean_ep_length", 0.0), step)
        self.tb.log_scalar("rollout/episodes", rollout_stats.get("episodes_finished", 0), step)
        self.tb.log_scalar("rollout/mean_reward", rollout_stats.get("mean_reward", 0.0), step)

        # --- Train / Losses ---
        self.tb.log_scalars("train", {
            "policy_loss": train_stats["policy_loss"],
            "value_loss": train_stats["value_loss"],
            "entropy": train_stats["entropy"],
            "approx_kl": train_stats.get("approx_kl", 0.0),
            "clip_fraction": train_stats.get("clip_fraction", 0.0),
            "explained_variance": train_stats.get("explained_variance", 0.0),
        }, step)

        # --- Optim ---
        self.tb.log_scalar("opt/learning_rate", train_stats["lr"], step)

        # --- Timing ---
        elapsed = time.time() - self.wall_start
        steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0.0
        self.tb.log_scalar("timing/steps_per_sec", steps_per_sec, step)
        self.tb.log_scalar("timing/fps", steps_per_sec, step)

        # --- Debug ---
        self.tb.log_scalar("debug/obs_norm_count", float(self.agent.model.obs_norm.count), step)

    def _should_stop_early(self) -> bool:
        """Early stopping si reward moyen > seuil sur la fenêtre."""
        if len(self.reward_history) < self.config.early_stop_window:
            return False
        return np.mean(self.reward_history) >= self.config.early_stop_threshold

    def run(self):
        """Boucle principale d'entraînement."""
        n_iters = self.config.total_timesteps // (self.config.n_envs * self.config.n_steps)
        print(f" Boucles d'entraînement : {n_iters}")

        for iteration in range(n_iters):
            self.iteration = iteration

            # --- Rollout ---
            rollout_stats = self.agent.rollout(self.env, self.config.n_steps)
            self.global_step += self.config.n_envs * self.config.n_steps

            # --- Entraînement ---
            obs, actions, log_probs_old, advantages, returns, values_old = self.agent.buffer.get_flat()
            train_stats = self.agent.update(
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                obs=obs,
                actions=actions,
                log_probs_old=log_probs_old,
                advantages=advantages,
                returns=returns,
                values_old=values_old,
            )

            # --- Logging ---
            self._log(rollout_stats, train_stats)

            # --- Reward history ---
            if rollout_stats["mean_ep_reward"] != 0:
                self.reward_history.append(rollout_stats["mean_ep_reward"])

            # --- Console print ---
            if iteration % self.config.log_interval == 0 or iteration == n_iters - 1:
                elapsed = time.time() - self.wall_start
                print(
                    f"[Iter {iteration:4d} | "
                    f"Step {self.global_step:>10,} | "
                    f"EpReward {rollout_stats['mean_ep_reward']:>7.1f} | "
                    f"MeanEp {np.mean(self.reward_history) if self.reward_history else -999:>7.1f} | "
                    f"Entropy {train_stats['entropy']:>6.3f} | "
                    f"LR {train_stats['lr']:.6f} | "
                    f"{elapsed:.0f}s"
                )

            # --- Checkpoint ---
            if iteration % self.config.save_freq == 0 and iteration > 0:
                ckpt_path = os.path.join(self.log_dir, f"ckpt_iter{iteration}")
                self.agent.save(ckpt_path)
                print(f"  → Checkpoint saved: {ckpt_path}")

            # --- Early stopping ---
            if self._should_stop_early():
                print(f"\n✓ Early stopping — reward moyen > {self.config.early_stop_threshold}")
                print(f"  sur {self.config.early_stop_window} épisodes")
                break

        # Save final
        final_path = os.path.join(self.log_dir, "final_model")
        self.agent.save(final_path)
        print(f"\n✓ Entraînement terminé")
        print(f"  Modèle final : {final_path}")
        self.tb.close()


def train_single_seed(env_id: str, config: PPOConfig, seed: int,
                      log_dir: str = None) -> dict:
    """Lance l'entraînement pour une graine."""
    setup_gpu(config)
    runner = TrainingRunner(config=config, env_id=env_id, seed=seed, log_dir=log_dir)
    runner.run()
    return {
        "seed": seed,
        "final_reward": np.mean(runner.reward_history) if runner.reward_history else 0.0,
        "total_steps": runner.global_step,
        "log_dir": runner.log_dir,
    }


def train_multiple_seeds(env_id: str, config: PPOConfig,
                         n_seeds: int = None,
                         base_log_dir: str = None) -> list:
    """Lance l'entraînement sur plusieurs seeds séquentiellement."""
    n_seeds = n_seeds or config.n_seeds
    base_log_dir = base_log_dir or "runs"

    results = []
    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed + 1}/{n_seeds}")
        print(f"{'='*60}\n")

        log_dir = os.path.join(base_log_dir, f"seed_{seed}")
        result = train_single_seed(env_id, config, seed=seed, log_dir=log_dir)
        results.append(result)

    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ MULTI-SEED")
    print(f"{'='*60}")
    for r in results:
        print(f"  Seed {r['seed']:2d} — reward: {r['final_reward']:7.1f} — steps: {r['total_steps']:,} — {r['log_dir']}")

    mean_reward = np.mean([r["final_reward"] for r in results])
    std_reward = np.std([r["final_reward"] for r in results])
    print(f"\n  Moyenne : {mean_reward:.1f} ± {std_reward:.1f}")
    return results

