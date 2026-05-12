#!/usr/bin/env python3
"""
Entry point évaluation PPO — BipedalWalker-v3.

Usage :
    python scripts/evaluate.py --checkpoint runs/seed_0/final_model
    python scripts/evaluate.py --checkpoint runs/seed_0/final_model --episodes 100 --render
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import load_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation PPO — BipedalWalker-v3")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Chemin vers le checkpoint (fichiers _actor.h5 etc.)")
    parser.add_argument("--env", type=str, default="BipedalWalker-v3",
                       help="Environnement")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Nombre d'épisodes (défaut: 100)")
    parser.add_argument("--render", action="store_true",
                       help="Afficher le rendu")
    parser.add_argument("--stochastic", action="store_true",
                       help="Actions stochastiques au lieu de déterministes")
    return parser.parse_args()


def main():
    args = parse_args()

    results = load_and_evaluate(
        checkpoint_path=args.checkpoint,
        env_id=args.env,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
