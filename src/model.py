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
    Les stats sont stockées à la fois en numpy (pour update CPU) et en
    tf.Variable (pour normalisation GPU sans transfert).
    """

    def __init__(self, shape: tuple, eps: float = 1e-5):
        self.eps = eps
        self.count = 0
        self.mean_np = np.zeros(shape, dtype=np.float32)
        self.var_np = np.ones(shape, dtype=np.float32)
        # Variables GPU pour normalisation zero-copy
        self.mean = tf.Variable(self.mean_np, trainable=False, dtype=tf.float32, name='obs_mean')
        self.var = tf.Variable(self.var_np, trainable=False, dtype=tf.float32, name='obs_var')

    def update(self, batch: np.ndarray) -> None:
        """Màj avec un batch d'observations (CPU)."""
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean_np
        total_count = self.count + batch_count

        new_mean = self.mean_np + delta * batch_count / total_count
        m_a = self.var_np * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean_np = new_mean
        self.var_np = np.maximum(new_var, self.eps)
        self.count = total_count

        # Sync vers GPU
        self.mean.assign(self.mean_np)
        self.var.assign(self.var_np)

    def normalize(self, obs):
        """Normalise une observation (ou batch) — GPU si tensor, CPU si numpy."""
        if isinstance(obs, np.ndarray):
            return (obs - self.mean_np) / np.sqrt(self.var_np + self.eps)
        # Tensor path (GPU)
        return (obs - self.mean) / tf.sqrt(self.var + self.eps)

    def __call__(self, obs):
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
        
        Note : log_std est cast au méme dtype que mu (utile pour mixed precision).
        std = exp(log_std), donc log_std peut prendre toute valeur réelle.
        Initialisé à 0 → std = 1 au départ.
        """
        mu = self.call(obs)
        log_std = tf.cast(self.log_std, mu.dtype)
        return mu, log_std

    def sample(self, obs: tf.Tensor) -> tf.Tensor:
        """Sample une action (utile pour rollout)."""
        mu, log_std = self.get_action_dist(obs)
        noise = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(log_std) * noise

    def log_prob(self, obs: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        """Log-probabilité log π(a|s) sous gaussienne diagonale."""
        mu, log_std = self.get_action_dist(obs)
        std = tf.exp(log_std)
        # Log-prob d'une gaussienne diagonale (pure TF, pas de numpy dans le graph)
        log_prob = -0.5 * (
            ((action - mu) / std)**2 +
            2.0 * log_std +
            tf.math.log(2.0 * tf.constant(np.pi, dtype=tf.float32))
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
        """Retourne (log_prob, value) pour un batch — utilisé pendant l'entraînement.
        Retourne des tensors (pas de .numpy()) pour rester sur GPU."""
        obs_t = tf.convert_to_tensor(self.obs_norm.normalize(obs), dtype=tf.float32)
        action_t = tf.convert_to_tensor(action, dtype=tf.float32)
        log_prob = self.actor.log_prob(obs_t, action_t)
        value = tf.squeeze(self.critic(obs_t), axis=-1)
        return log_prob, value

    def summary(self):
        print("=== Actor ===")
        self.actor.summary()
        print("\n=== Critic ===")
        self.critic.summary()

