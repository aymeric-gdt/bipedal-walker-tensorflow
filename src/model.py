"""
Réseaux Actor et Critic séparés pour PPO sur BipedalWalker-v3.

Architecture :
  - Actor  : obs → 128 → 128 → μ (action mean, tanh borné)
  - Critic : obs → 256 → 256 → V(s)

log_std est un tf.Variable trainable (shape 4,), pas une sortie du réseau.
Les deux réseaux sont complètement séparés — pas de tronc commun.

Sources :
  - "What Matters In On-Policy RL?" (arXiv:2006.05990)
  - "Honey, I Shrunk The Actor" (arXiv:2102.11893)
  - Engstrom et al. 2020
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RunningNorm:
    """Normalisation des observations : running mean/std.

    Màj incrémentale du mean et variance (Welford online algorithm).
    """

    def __init__(self, shape: tuple, eps: float = 1e-5):
        self.eps = eps
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)

    def update(self, batch: np.ndarray) -> None:
        """Màj avec un batch d'observations."""
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, self.eps)
        self.count = total_count

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalise une observation (ou batch)."""
        return (obs - self.mean) / np.sqrt(self.var + self.eps)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.normalize(obs)


class ActorNetwork(keras.Model):
    """Actor : calcule μ(a|s) pour une distribution gaussienne diagonale."""

    def __init__(self, action_dim: int = 4, hidden: tuple = (128, 128),
                 activation: str = "tanh", name: str = "actor"):
        super().__init__(name=name)
        self.hidden_units = hidden
        self.action_dim = action_dim
        self.activation = activation

        self.layers_ = [
            layers.Dense(h, activation=activation, kernel_initializer="orthogonal")
            for h in hidden
        ]
        self.mu_out = layers.Dense(action_dim, activation="tanh",
                                   kernel_initializer="orthogonal",
                                   name="mu")

        # log_std trainable via add_weight (Keras 3 compatible)
        self.log_std = self.add_weight(
            shape=(action_dim,),
            initializer="zeros",
            trainable=True,
            name="log_std",
            dtype=tf.float32,
        )

    def call(self, obs: tf.Tensor) -> tf.Tensor:
        """Retourne μ(a|s) — action mean, tanh borné."""
        x = obs
        for layer in self.layers_:
            x = layer(x)
        return self.mu_out(x)

    def get_action_dist(self, obs: tf.Tensor):
        """Retourne (mu, log_std) — pour le calcul de log_prob.
        
        Note : log_std est renvoyé directement (pas de contrainte ici).
        std = exp(log_std), donc log_std peut prendre toute valeur réelle.
        Initialisé à 0 → std = 1 au départ.
        """
        mu = self.call(obs)
        return mu, self.log_std

    def sample(self, obs: tf.Tensor) -> tf.Tensor:
        """Sample une action (utile pour rollout)."""
        mu, log_std = self.get_action_dist(obs)
        noise = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(log_std) * noise

    def log_prob(self, obs: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        """Log-probabilité log π(a|s) sous gaussienne diagonale."""
        mu, log_std = self.get_action_dist(obs)
        std = tf.exp(log_std)
        # Log-prob d'une gaussienne diagonale
        log_prob = -0.5 * (
            ((action - mu) / std)**2 +
            2 * log_std +
            np.log(2 * np.pi)
        )
        return tf.reduce_sum(log_prob, axis=-1)


class CriticNetwork(keras.Model):
    """Critic : V(s) — value function estimate."""

    def __init__(self, hidden: tuple = (256, 256),
                 activation: str = "tanh", name: str = "critic"):
        super().__init__(name=name)
        self.hidden_units = hidden
        self.activation = activation

        self.layers_ = [
            layers.Dense(h, activation=activation, kernel_initializer="orthogonal")
            for h in hidden
        ]
        self.v_out = layers.Dense(1, kernel_initializer="orthogonal", name="v")

    def call(self, obs: tf.Tensor) -> tf.Tensor:
        """Retourne V(s)."""
        x = obs
        for layer in self.layers_:
            x = layer(x)
        return self.v_out(x)


class PPOModel:
    """Modèle PPO complet : Actor + Critic séparés + RunningNorm."""

    def __init__(self, obs_dim: int = 24, action_dim: int = 4,
                 actor_hidden: tuple = (128, 128),
                 critic_hidden: tuple = (256, 256),
                 activation: str = "tanh",
                 log_std_init: float = 0.0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.obs_norm = RunningNorm((obs_dim,))

        self.actor = ActorNetwork(
            action_dim=action_dim,
            hidden=actor_hidden,
            activation=activation
        )
        self.critic = CriticNetwork(
            hidden=critic_hidden,
            activation=activation
        )

        # Build pour que les variables soient créées
        dummy_obs = tf.zeros((1, obs_dim), dtype=tf.float32)
        self.actor(dummy_obs)
        self.critic(dummy_obs)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return self.obs_norm.normalize(obs)

    def update_obs_norm(self, obs: np.ndarray) -> None:
        """Màj les stats de normalisation (sur ancien batch)."""
        self.obs_norm.update(obs)

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Retourne une action (numpy)."""
        obs_t = tf.convert_to_tensor(self.obs_norm.normalize(obs), dtype=tf.float32)
        if deterministic:
            action = self.actor.call(obs_t)
        else:
            action = self.actor.sample(obs_t)
        return action.numpy()

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        """Retourne V(s) pour une observation (numpy)."""
        obs_t = tf.convert_to_tensor(self.obs_norm.normalize(obs), dtype=tf.float32)
        return self.critic(obs_t).numpy().flatten()

    def get_log_prob_and_value(self, obs: np.ndarray,
                                action: np.ndarray):
        """Retourne (log_prob, value) pour un batch — utilisé pendant l'entraînement."""
        obs_t = tf.convert_to_tensor(self.obs_norm.normalize(obs), dtype=tf.float32)
        action_t = tf.convert_to_tensor(action, dtype=tf.float32)
        log_prob = self.actor.log_prob(obs_t, action_t)
        value = self.critic(obs_t).numpy().flatten()
        return log_prob.numpy(), value

    def summary(self):
        print("=== Actor ===")
        self.actor.summary()
        print("\n=== Critic ===")
        self.critic.summary()

