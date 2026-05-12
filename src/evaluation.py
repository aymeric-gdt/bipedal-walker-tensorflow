"""
Évaluation d'un agent PPO entraîné sur BipedalWalker-v3.

Inclut :
  - Évaluation déterministe (action moyenne)
  - Rendu visuel
  - Stats : mean, std, min, max reward
"""

import numpy as np
import gymnasium as gym

from src.config import PPOConfig
from src.agent import PPOAgent


def evaluate_agent(agent: PPOAgent,
                  env_id: str = "BipedalWalker-v3",
                  n_episodes: int = 100,
                  render: bool = False,
                  deterministic: bool = True) -> dict:
    """Évalue un agent sur n_episodes.

    Args:
        agent: Agent PPO (doit avoir .model chargé)
        env_id: Environnement Gymnasium
        n_episodes: Nombre d'épisodes
        render: Afficher le rendu
        deterministic: Utiliser action déterministe (μ) au lieu de sampler

    Returns:
        dict avec rewards, lengths, stats
    """
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)

    rewards = []
    lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            action = agent.model.get_action(
                obs[np.newaxis, ...],
                deterministic=deterministic
            )[0]

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1

            if render:
                env.render()

        rewards.append(ep_reward)
        lengths.append(ep_length)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{n_episodes} — reward: {ep_reward:.1f}, length: {ep_length}")

    env.close()

    rewards = np.array(rewards)
    lengths = np.array(lengths)

    return {
        "rewards": rewards,
        "lengths": lengths,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(rewards > 300)),  # solved threshold
    }


def print_eval_report(results: dict):
    """Affiche un rapport d'évaluation."""
    print("\n" + "=" * 50)
    print("RAPPORT D'ÉVALUATION")
    print("=" * 50)
    print(f"  Episodes:         {len(results['rewards'])}")
    print(f"  Mean reward:      {results['mean_reward']:8.1f}")
    print(f"  Std reward:       {results['std_reward']:8.1f}")
    print(f"  Min reward:       {results['min_reward']:8.1f}")
    print(f"  Max reward:       {results['max_reward']:8.1f}")
    print(f"  Mean length:      {results['mean_length']:8.1f}")
    print(f"  Success rate (>300): {results['success_rate']*100:6.1f}%")
    print("=" * 50)


def load_and_evaluate(checkpoint_path: str,
                     env_id: str = "BipedalWalker-v3",
                     n_episodes: int = 100,
                     **eval_kwargs) -> dict:
    """Charge un checkpoint et évalue.

    Usage:
        results = load_and_evaluate("runs/seed_0/final_model")
    """
    config = PPOConfig()
    agent = PPOAgent(obs_dim=24, action_dim=4, config=config)
    agent.load(checkpoint_path)

    results = evaluate_agent(agent, env_id=env_id,
                             n_episodes=n_episodes, **eval_kwargs)
    print_eval_report(results)
    return results

