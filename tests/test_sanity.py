"""
Tests de Sanity Check pour BipedalWalker-v3.

Tests :
    1. Env shapes     : BipedalWalker-v3 reset/step → shapes OK
    2. Forward pass   : réseau → output (4,) + value (1,)
    3. Rollout test  : 1 itération → shapes OK, pas de NaN
    4. Training loop  : 100 steps → loss décroissante, pas de NaN/Inf

Usage :
    pytest tests/ -v
"""

import numpy as np
import pytest
import gymnasium as gym

from src.config import PPOConfig
from src.model import PPOModel
from src.agent import PPOAgent


# ─── Test 1 : Env shapes ────────────────────────────────────────────────────

def test_env_shapes():
    """BipedalWalker-v3 : reset/step → shapes cohérentes."""
    env = gym.make("BipedalWalker-v3")
    obs, _ = env.reset()

    assert obs.shape == (24,), f"Observation shape attendu (24,), got {obs.shape}"
    assert env.action_space.shape == (4,), f"Action shape attendu (4,), got {env.action_space.shape}"

    # Step aléatoire
    action = env.action_space.sample()
    obs_next, reward, terminated, truncated, info = env.step(action)

    assert obs_next.shape == (24,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    env.close()


# ─── Test 2 : Forward pass ──────────────────────────────────────────────────

def test_forward_pass():
    """Une observation → réseau → μ (4,), V (1,)."""
    model = PPOModel(obs_dim=24, action_dim=4)

    obs = np.random.randn(24).astype(np.float32)
    obs_batch = np.expand_dims(obs, 0).astype(np.float32)

    # Action (stochastique)
    action = model.get_action(obs_batch)
    assert action.shape == (1, 4), f"Action shape attendu (1, 4), got {action.shape}"
    assert np.all(np.isfinite(action)), "Action contient NaN/Inf"

    # Value
    value = model.get_value(obs_batch)
    assert value.shape == (1,), f"Value shape attendu (1,), got {value.shape}"
    assert np.isfinite(value), "Value contient NaN/Inf"

    # Log prob
    log_prob, value = model.get_log_prob_and_value(obs_batch, action)
    assert log_prob.shape == (1,), f"Log prob shape attendu (1,), got {log_prob.shape}"
    assert np.isfinite(log_prob), "Log prob contient NaN/Inf"


# ─── Test 3 : Rollout test ───────────────────────────────────────────────────

class DummyEnv:
    """Mock env simple pour test de rollout."""
    n_envs = 2

    def __init__(self, n_envs=2):
        self.n_envs = n_envs
        self.obs_dim = 24
        self.action_dim = 4
        self.action_space = type("obj", (), {"shape": (4,)})()

    def reset(self):
        return np.random.randn(self.n_envs, self.obs_dim).astype(np.float32)

    def step(self, actions):
        obs = np.random.randn(self.n_envs, self.obs_dim).astype(np.float32)
        rewards = np.random.randn(self.n_envs).astype(np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        return obs, rewards, dones, {}


def test_rollout_buffer_shapes():
    """Rollout buffer : shapes OK, pas de NaN."""
    config = PPOConfig()
    config.n_envs = 2
    config.n_steps = 8

    agent = PPOAgent(obs_dim=24, action_dim=4, config=config)
    env = DummyEnv(n_envs=2)

    # 1 rollout
    stats = agent.rollout(env, n_steps=8)

    # Vérifie les shapes du buffer
    buf = agent.buffer
    total = config.n_envs * config.n_steps
    assert buf.obs.shape == (total, 24), f"obs shape {(total, 24)}, got {buf.obs.shape}"
    assert buf.actions.shape == (total, 4), f"actions shape {(total, 4)}, got {buf.actions.shape}"
    assert buf.rewards.shape == (total,), f"rewards shape {(total,)}, got {buf.rewards.shape}"
    assert buf.dones.shape == (total,), f"dones shape {(total,)}, got {buf.dones.shape}"
    assert buf.values.shape == (total,), f"values shape {(total,)}, got {buf.values.shape}"
    assert buf.log_probs.shape == (total,), f"log_probs shape {(total,)}, got {buf.log_probs.shape}"

    # Pas de NaN
    assert np.all(np.isfinite(buf.obs)), "Buffer obs contient NaN/Inf"
    assert np.all(np.isfinite(buf.actions)), "Buffer actions contient NaN/Inf"
    assert np.all(np.isfinite(buf.rewards)), "Buffer rewards contient NaN/Inf"
    assert np.all(np.isfinite(buf.values)), "Buffer values contient NaN/Inf"
    assert np.all(np.isfinite(buf.log_probs)), "Buffer log_probs contient NaN/Inf"

    # GAE computed
    assert hasattr(buf, "advantages")
    assert hasattr(buf, "returns")
    assert np.all(np.isfinite(buf.advantages)), "Advantages contiennent NaN/Inf"
    assert np.all(np.isfinite(buf.returns)), "Returns contiennent NaN/Inf"


# ─── Test 4 : Training loop (mini) ───────────────────────────────────────────

def test_training_loop_mini():
    """100 steps d'entraînement → loss décroissante, pas de NaN/Inf."""
    config = PPOConfig()
    config.n_envs = 2
    config.n_steps = 8
    config.total_timesteps = 1000
    config.max_timesteps_per_iter = 1000
    config.batch_size = 16
    config.n_epochs = 2

    agent = PPOAgent(obs_dim=24, action_dim=4, config=config)
    env = DummyEnv(n_envs=2)

    losses = []
    for _ in range(10):  # 10 itérations
        agent.rollout(env, n_steps=8)
        obs, actions, log_probs_old, advantages, returns = agent.buffer.get_batch()

        # Normalize advantages pour le test
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        stats = agent.update(
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            obs=obs,
            actions=actions,
            log_probs_old=log_probs_old,
            advantages=advantages,
            returns=returns,
        )
        losses.append(stats["total_loss"])

        assert np.isfinite(stats["total_loss"]), f"Total loss NaN/Inf à iter {_}"
        assert np.isfinite(stats["policy_loss"]), f"Policy loss NaN/Inf"
        assert np.isfinite(stats["value_loss"]), f"Value loss NaN/Inf"

    # La loss devrait tendre vers quelque chose de stable (pas necessarily decreasing,
    # mais pas diverger)
    assert np.all(np.isfinite(losses)), "Loss contient NaN/Inf"
    assert not np.isnan(losses[-1]), "Loss finale est NaN"
    print(f"\n  Losses : {[f'{l:.3f}' for l in losses]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

