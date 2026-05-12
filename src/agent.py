"""
Agent PPO : implémentation complète pour BipedalWalker-v3.

Composants :
  - RolloutBuffer : stockage des transitions
  - PPOAgent : logique PPO (GAE, clipped objective, value clipping, entropy bonus)

Sources :
  - Schulman et al. 2017 (arXiv:1707.06347)
  - Engstrom et al. 2020 (arXiv:2005.12729)
  - OpenAI Baselines ppo2
"""

import numpy as np
import tensorflow as tf
from typing import Tuple

from src.config import PPOConfig
from src.model import PPOModel


class RolloutBuffer:
    """Buffer circulaire pour le rollout PPO.

    Stocke : obs (normalisées), actions, rewards, dones, values, log_probs.
    Calcule : GAE avantages + returns.
    """

    def __init__(self, n_envs: int, n_steps: int, obs_dim: int, action_dim: int,
                 gamma: float, lam: float, rew_clip: Tuple[float, float]):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.rew_clip = rew_clip

        self.total_size = n_envs * n_steps
        self._setup_buffers()

    def _setup_buffers(self):
        # obs stockées NORMALISÉES (c'est ce qui est passé au réseau)
        self.obs = np.zeros((self.n_steps, self.n_envs, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

        self.advantages = None
        self.returns = None

    def store_step(self, step: int, obs_norm: np.ndarray, actions: np.ndarray,
                   rewards: np.ndarray, dones: np.ndarray,
                   values: np.ndarray, log_probs: np.ndarray):
        """Store un step complet (tous envs d'un coup)."""
        self.obs[step] = obs_norm
        self.actions[step] = actions
        self.rewards[step] = np.clip(rewards, self.rew_clip[0], self.rew_clip[1])
        self.dones[step] = dones
        self.values[step] = values
        self.log_probs[step] = log_probs

    def compute_gae_returns(self, last_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule GAE avantages et returns.

        last_values : (n_envs,) — valeurs V(s_T) pour bootstrapping.
        Retourne :
          advantages : (n_steps * n_envs,)
          returns    : (n_steps * n_envs,) — pour value loss
        """
        advantages = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

        gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_values
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + self.values
        return advantages.reshape(-1), returns.reshape(-1)

    def get_flat(self):
        """Retourne les données aplaties pour l'entraînement.

        Returns : (obs, actions, log_probs_old, advantages, returns, values_old)
        """
        obs_flat = self.obs.reshape(-1, self.obs_dim)
        actions_flat = self.actions.reshape(-1, self.action_dim)
        log_probs_flat = self.log_probs.reshape(-1)
        adv_flat = self.advantages  # déjà (total_size,)
        ret_flat = self.returns     # déjà (total_size,)
        values_flat = self.values.reshape(-1)
        return obs_flat, actions_flat, log_probs_flat, adv_flat, ret_flat, values_flat

    def reset(self):
        """Reset le buffer."""
        self._setup_buffers()


class PPOAgent:
    """Agent PPO complet.

    Inclut :
      - Clipped surrogate objective (ε=0.2)
      - Value function loss with clipping (Engstrom et al.)
      - Entropy bonus (coefficient fixe)
      - Adam avec LR annealing
      - Gradient clipping (global norm = 0.5)
    """

    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

        self.model = PPOModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden=config.actor_hidden,
            critic_hidden=config.critic_hidden,
            activation=config.activation,
            log_std_init=config.log_std_init,
        )

        # Optimizers séparés pour actor et critic
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.lr,
            epsilon=config.adam_epsilon,
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.lr,
            epsilon=config.adam_epsilon,
        )

        # Buffer
        self.buffer = RolloutBuffer(
            n_envs=config.n_envs,
            n_steps=config.n_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            gamma=config.gamma,
            lam=config.lam,
            rew_clip=config.rew_clip,
        )

        self.iteration = 0
        # Obs courante persistée entre rollouts
        self._last_obs = None

    def _get_lr(self) -> float:
        """Learning rate avec annealing linéaire."""
        if self.config.lr_annealing:
            total_iters = self.config.total_timesteps // (self.config.n_envs * self.config.n_steps)
            progress = self.iteration / max(total_iters, 1)
            return self.config.lr * max(0.0, 1.0 - progress)
        return self.config.lr

    def rollout(self, envs, n_steps: int) -> dict:
        """Collecte n_steps transitions avec les envs parallèles.

        IMPORTANT : ne reset PAS les envs entre les rollouts — l'état est
        maintenu en continu. Reset initial uniquement au premier appel.
        """
        if self._last_obs is None:
            self._last_obs = envs.reset()

        obs_batch = self._last_obs
        episode_rewards = []
        episode_lengths = []
        ep_reward = np.zeros(envs.num_envs, dtype=np.float32)
        ep_length = np.zeros(envs.num_envs, dtype=np.float32)

        for step in range(n_steps):
            # Normaliser + mettre à jour les stats
            self.model.obs_norm.update(obs_batch)
            obs_norm = self.model.obs_norm.normalize(obs_batch)

            # Actions stochastiques
            obs_t = tf.convert_to_tensor(obs_norm, dtype=tf.float32)
            actions = self.model.actor.sample(obs_t).numpy()

            # Log probs et values sur obs normalisées
            log_probs = self.model.actor.log_prob(obs_t, tf.convert_to_tensor(actions, dtype=tf.float32)).numpy()
            values = self.model.critic(obs_t).numpy().flatten()

            obs_next, rewards, dones, infos = envs.step(actions)

            # Stats épisode
            ep_reward += rewards
            ep_length += 1
            for env_idx in range(envs.num_envs):
                if dones[env_idx]:
                    episode_rewards.append(float(ep_reward[env_idx]))
                    episode_lengths.append(int(ep_length[env_idx]))
                    ep_reward[env_idx] = 0.0
                    ep_length[env_idx] = 0.0

            # Store dans buffer (obs normalisées)
            self.buffer.store_step(step, obs_norm, actions, rewards, dones, values, log_probs)

            obs_batch = obs_next

        self._last_obs = obs_batch

        # Bootstrap last values
        self.model.obs_norm.update(obs_batch)
        obs_norm_last = self.model.obs_norm.normalize(obs_batch)
        last_values = self.model.critic(
            tf.convert_to_tensor(obs_norm_last, dtype=tf.float32)
        ).numpy().flatten()

        # Compute GAE
        advantages, returns = self.buffer.compute_gae_returns(last_values)
        self.buffer.advantages = advantages
        self.buffer.returns = returns

        stats = {
            "mean_reward": float(np.mean(rewards)),
            "mean_length": float(np.mean(ep_length)),
            "episodes_finished": len(episode_rewards),
            "mean_ep_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "mean_ep_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        }
        return stats

    @tf.function
    def _train_step(self, obs: tf.Tensor, actions: tf.Tensor,
                    log_probs_old: tf.Tensor,
                    advantages: tf.Tensor,
                    returns: tf.Tensor,
                    values_old: tf.Tensor,
                    lr: tf.Tensor) -> dict:
        """Une passe d'entraînement PPO (tf.function pour performance)."""
        # Normalise advantages (par batch)
        if self.config.normalize_advantage:
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        actor_vars = self.model.actor.trainable_variables
        critic_vars = self.model.critic.trainable_variables

        # --- Actor update ---
        with tf.GradientTape() as tape_actor:
            mu, log_std = self.model.actor.get_action_dist(obs)
            std = tf.exp(log_std)

            log_prob_new = tf.reduce_sum(
                -0.5 * (((actions - mu) / std) ** 2 + 2.0 * log_std + tf.math.log(2.0 * np.pi)),
                axis=-1
            )

            ratio = tf.exp(log_prob_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio,
                                      1.0 - self.config.clip_ratio,
                                      1.0 + self.config.clip_ratio) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Entropy : H = 0.5 * (1 + log(2π σ²)) per dim, summed over action dims
            entropy = tf.reduce_mean(
                tf.reduce_sum(0.5 * (1.0 + tf.math.log(2.0 * np.pi * std ** 2)), axis=-1)
            )
            actor_loss = policy_loss - self.config.entropy_coef * entropy

        actor_grads = tape_actor.gradient(actor_loss, actor_vars)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.config.max_grad_norm)
        self.actor_optimizer.learning_rate.assign(lr)
        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))

        # --- Critic update ---
        with tf.GradientTape() as tape_critic:
            values_new = tf.squeeze(self.model.critic(obs), axis=-1)

            # Value clipping (Engstrom et al.)
            v_clipped = values_old + tf.clip_by_value(
                values_new - values_old,
                -self.config.value_clip,
                self.config.value_clip
            )
            v_loss1 = tf.square(values_new - returns)
            v_loss2 = tf.square(v_clipped - returns)
            critic_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss1, v_loss2)) * self.config.value_coef

        critic_grads = tape_critic.gradient(critic_loss, critic_vars)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.config.max_grad_norm)
        self.critic_optimizer.learning_rate.assign(lr)
        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))

        return {
            "policy_loss": policy_loss,
            "value_loss": critic_loss,
            "entropy": entropy,
            "log_std_mean": tf.reduce_mean(log_std),
        }

    def update(self, batch_size: int, n_epochs: int,
               obs: np.ndarray, actions: np.ndarray,
               log_probs_old: np.ndarray,
               advantages: np.ndarray,
               returns: np.ndarray,
               values_old: np.ndarray) -> dict:
        """Effectue n_epochs de mise à jour PPO sur le rollout buffer."""
        n_samples = obs.shape[0]
        indices = np.arange(n_samples)

        policy_losses, value_losses, entropies, log_std_means = [], [], [], []

        lr = tf.constant(self._get_lr(), dtype=tf.float32)

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                idx = indices[start:start + batch_size]
                losses = self._train_step(
                    tf.constant(obs[idx], dtype=tf.float32),
                    tf.constant(actions[idx], dtype=tf.float32),
                    tf.constant(log_probs_old[idx], dtype=tf.float32),
                    tf.constant(advantages[idx], dtype=tf.float32),
                    tf.constant(returns[idx], dtype=tf.float32),
                    tf.constant(values_old[idx], dtype=tf.float32),
                    lr,
                )
                policy_losses.append(float(losses["policy_loss"]))
                value_losses.append(float(losses["value_loss"]))
                entropies.append(float(losses["entropy"]))
                log_std_means.append(float(losses["log_std_mean"]))

        self.iteration += 1

        # Diagnostic post-update
        raw_ls = self.model.actor.log_std.numpy()
        print(f"    [DIAG] log_std={np.round(raw_ls, 4)}  entropy={np.mean(entropies):.4f}  "
              f"policy_loss={np.mean(policy_losses):.4f}  value_loss={np.mean(value_losses):.4f}")

        return {
            "total_loss": float(np.mean(policy_losses)) + float(np.mean(value_losses)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "lr": float(lr),
        }

    def save(self, path: str):
        """Sauvegarde le modèle complet."""
        self.model.actor.save_weights(f"{path}_actor.weights.h5")
        self.model.critic.save_weights(f"{path}_critic.weights.h5")
        np.savez(f"{path}_obs_norm.npz",
                 mean=self.model.obs_norm.mean,
                 var=self.model.obs_norm.var,
                 count=self.model.obs_norm.count)

    def load(self, path: str):
        """Charge le modèle complet."""
        self.model.actor.load_weights(f"{path}_actor.weights.h5")
        self.model.critic.load_weights(f"{path}_critic.weights.h5")
        data = np.load(f"{path}_obs_norm.npz")
        self.model.obs_norm.mean = data["mean"]
        self.model.obs_norm.var = data["var"]
        self.model.obs_norm.count = data["count"]
