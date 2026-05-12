#!/usr/bin/env python3
"""
Entry point entraînement PPO — BipedalWalker-v3.

Usage :
    python scripts/train.py                      # 1 seed, config par défaut
    python scripts/train.py --seeds 3            # 3 seeds
    python scripts/train.py --env BipedalWalker-v3 --total-steps 5_000_000
"""
import argparse
import sys
import os

# Ajout du parent au path pour imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PPOConfig
from src.training import train_single_seed, train_multiple_seeds


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement PPO — BipedalWalker-v3")
    parser.add_argument("--env", type=str, default="BipedalWalker-v3",
                       help="Environnement Gymnasium")
    parser.add_argument("--seeds", type=int, default=1,
                       help="Nombre de seeds (défaut: 1)")
    parser.add_argument("--total-steps", type=int, default=None,
                       help="Total steps (défaut: config.default)")
    parser.add_argument("--n-envs", type=int, default=None,
                       help="Nombre d'envs parallèles (défaut: 8)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (défaut: 3e-4)")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Répertoire de logs")
    parser.add_argument("--seed", type=int, default=0,
                       help="Graine de base (défaut: 0)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Config
    config = PPOConfig()
    if args.total_steps is not None:
        config.total_timesteps = args.total_steps
        config.max_timesteps_per_iter = args.total_steps
    if args.n_envs is not None:
        config.n_envs = args.n_envs
    if args.lr is not None:
        config.lr = args.lr

    print(f"Config : {config.n_envs} envs, {config.total_timesteps:,} steps, "
          f"lr={config.lr}, seeds={args.seeds}")

    if args.seeds == 1:
        result = train_single_seed(
            env_id=args.env,
            config=config,
            seed=args.seed,
            log_dir=args.log_dir,
        )
        print(f"\nRésultat : reward={result['final_reward']:.1f}, "
              f"steps={result['total_steps']:,}")
    else:
        results = train_multiple_seeds(
            env_id=args.env,
            config=config,
            n_seeds=args.seeds,
            base_log_dir=args.log_dir,
        )


if __name__ == "__main__":
    main()

