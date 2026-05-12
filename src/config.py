"""
Hyperparamètres PPO pour BipedalWalker-v3.
Sources : Engstrom et al. 2020, OpenAI Baselines, Stable-Baselines3, sgoodfriend/ppo-BipedalWalker-v3.
"""

from typing import Tuple
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """Hyperparamètres PPO — BipedalWalker-v3."""

    # --- Rollout ---
    n_envs: int = 8                  # nombre d'envs parallèles (stable-baselines3 default: 1)
    n_steps: int = 2048              # steps par env avant mise à jour
    # Total batch par update = n_envs * n_steps = 16384

    # --- Optimisation ---
    n_epochs: int = 10               # nombre de passes SGD sur le rollout buffer
    batch_size: int = 512             # mini-batch size (plus gros = meilleur GPU saturation)
    lr: float = 3e-4                 # learning rate initial (Adam)
    lr_annealing: bool = True         # linear decay vers 0
    adam_epsilon: float = 1e-5        # epsilon pour Adam (plus grand que défaut TF)

    # --- PPO clipping ---
    clip_ratio: float = 0.2          # epsilon de clipping PPO (Schulman et al. 2017)
    value_clip: float = 0.2           # clipping de la value function (Engstrom et al.)

    # --- GAE ---
    gamma: float = 0.99              # discount factor
    lam: float = 0.95                # GAE lambda

    # --- Loss ---
    value_coef: float = 0.5          # coefficient de la value loss
    entropy_coef: float = 0.01      # coefficient d'entropie (fixe — 0.01 est trop élevé)
    max_grad_norm: float = 0.5        # global gradient clipping

    # --- Normalisation ---
    normalize_obs: bool = True        # running mean/std sur observations
    normalize_advantage: bool = True  # normaliser les avantages avant PG objective
    rew_clip: Tuple[float, float] = (-10.0, 10.0)  # reward clipping

    # --- Réseau ---
    actor_hidden: Tuple[int, ...] = (128, 128)   # actor: 2×128 (plus étroit)
    critic_hidden: Tuple[int, ...] = (256, 256)  # critic: 2×256 (plus large — Honey et al.)
    activation: str = "tanh"
    log_std_init: float = 0.0        # init de log_std (scalar, sera broadcasté)

    # --- Entraînement ---
    total_timesteps: int = 10_000_000  # 10M steps (config HF de référence)
    max_timesteps_per_iter: int = 10_000_000  # pour LR annealing (total_steps / n_iters per env)
    eval_freq: int = 50              # eval tous les n itérations
    eval_episodes: int = 100         # nombre d'épisodes pour l'évaluation
    save_freq: int = 50              # checkpoint toutes les n itérations
    early_stop_threshold: float = 300.0  # reward moyen pour early stopping
    early_stop_window: int = 100     # fenêtre pour early stopping

    # --- Divers ---
    seed: int = 0                   # graine principale
    n_seeds: int = 3                # nombre de seeds à runner
    log_interval: int = 1            # intervalle de logging TensorBoard

    # --- GPU / Performance ---
    use_mixed_precision: bool = True  # float16 forward pass pour 2-3x speedup sur RTX/Ampere
    gpu_memory_growth: bool = True    # évite d'allouer toute la VRAM d'un coup
    xla_jit: bool = True              # XLA compilation pour les @tf.function


# Singleton
config = PPOConfig()
