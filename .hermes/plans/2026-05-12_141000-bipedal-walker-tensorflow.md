# BipedalWalker TensorFlow — Plan d'implémentation

## Objectif

Entraîner un agent (PPO préféré) pour résoudre l'environnement `BipedalWalker-v3` de Gymnasium (Box2D), en pur TensorFlow.

---

## Contexte / Hypothèses

- Environnement : `BipedalWalker-v3` (observation continue 24D, action continue 4D)
- Contraintes :
  - GPU disponible (CUDA) pour l'entraînement
  - TensorFlow 2.x
  - Gymnasium (pas OpenAI Gym)
- Terrain : `stagger` (par défaut), `hardcore` en extension
- Métrique de succès : reward > 300 sur 100 épisodes consécutifs

---

## Approche proposée

### Algorithme : PPO (Proximal Policy Optimization)

Pourquoi PPO :
- state-of-the-art pour les problèmes d'action continue
- stable en pratique (clipped objective)
- bonnes implémentations TensorFlow disponibles (TF-Agents, stable-baselines3)
- évite d'avoir à réinventer la roue

Alternatives envisagées :
| Algorithme | Pros | Cons |
|---|---|---|
| **PPO** | Stable, benchmark robuste | Plus lourd qu'A2C |
| SAC | Bonus exploration, continue | Plus complexe à tuner |
| DDPG | Simple, déterministe | Instable, pas de clipped trust region |
| A2C | Léger, synchrone | Moins stable que PPO |

---

## Architecture du projet

```
bipedal-walker-tensorflow/
├── .hermes/
│   └── plans/
├── src/
│   ├── __init__.py
│   ├── config.py          # Hyperparamètres
│   ├── model.py           # Réseau de neurones (actor + critic)
│   ├── agent.py           # Classe PPO principale
│   ├── training.py        # Boucle d'entraînement
│   └── evaluation.py      # Inference + rendu
├── environments/
│   └── dummy.py           # Wrappers custom si besoin
├── data/
│   └── logs/              # TensorBoard logs
├── scripts/
│   ├── train.py           # Entry point entraînement
│   └── evaluate.py        # Entry point évaluation
├── requirements.txt
└── README.md
```

---

## Recherche — Sources et résultats existants

### Résultats de référence

| Repo | Algo | Stars | Steps to 300 | Notes |
|---|---|---|---|---|
| [leonjovanovic/drl-ppo-bipedal-walker](https://github.com/leonjovanovic/drl-ppo-bipedal-walker) | PPO (PyTorch) | 6 | ~700K (single env) | Best single ep 320.5 |
| [BrutFab/ppo_BipedalWalker_v3](https://github.com/BrutFab/ppo_BipedalWalker_v3) | PPO | 9 | — | HuggingFace integration |
| [aryannzzz/ppo-walker-rl](https://github.com/aryannzzz/ppo-walker-rl) | PPO | 7 | — | PyBullet, ablations |
| [Kyziridis/BipedalWalker-v2](https://github.com/Kyziridis/BipedalWalker-v2) | A2C/DQN | 27 | — | old Gym, bugs |
| [sgoodfriend/ppo-BipedalWalker-v3](https://huggingface.co/sgoodfriend/ppo-BipedalWalker-v3) | PPO | — | **10M** (16 envs) | **Config de référence** — 16 envs, n_steps 2048, ent_coef 0.001, lr 2.5e-4 linear decay, normalize true |

### Améliorations critiques — "Implementation Matters in Deep Policy Gradients" (Engstrom et al.)

> Source : [arXiv:2005.12729](https://arxiv.org/pdf/2005.12729.pdf)

Ces améliorations ne sont PAS dans le papier PPO original mais font une différence énorme en pratique :

| Improvement | Description | Impact |
|---|---|---|
| **Value Function Loss Clipping** | `value_clipped = V_old + clamp(V_new - V_old, -clip, +clip)` | Stabilise le value learning |
| **Adam LR Annealing** | Linear decay du learning rate | Évite l'overshoot en fin d'entraînement |
| **Global Gradient Clipping** | `max_grad_norm = 0.5` | Empêche les explosions de gradient |
| **Observation Normalization** | Running mean/std sur les observations | Meilleure généralisation |
| **Observation Clipping** | Clip observations à `[-10, 10]` | Robustesse aux outliers |
| **Reward Scaling + Clipping** | Clip rewards à `[-10, 10]` | Réduit variance des gradients |
| **Tanh activations** | `tanh` partout dans le réseau | Bounded activations, stable |

### Améliorations — OpenAI baselines PPO

> Source : [openai/baselines/ppo2](https://github.com/openai/baselines/tree/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2)

| Improvement | Description |
|---|---|
| **Separate networks** | Actor et Critic ont des hidden layers séparés (NE PAS partager) |
| **GAE** | Generalized Advantage Estimation (λ=0.95) |
| **Advantage Normalization** | Normaliser les avantages avantPG objective |
| **Adam ε** | Utiliser `ε=1e-5` dans Adam (plus grand que défaut TF) |
| **Entropy bonus** | Ajout entropie à la loss totale |

### PPO paper original : arXiv:1707.06347 (Schulman et al.)

L'algorithme original utilise `clip_ratio ε=0.2` (donc garder 0.2).

---

## Étapes détaillées

### Étape 1 — Scaffolding projet

- Créer la structure de fichiers ci-dessus
- `requirements.txt` : tensorflow, gymnasium[box2d], tensorflow-probability, numpy, matplotlib
- **Note multi-envs** : pour 8–16 envs parallèles, TF-Agents fournit `TF2MultiAgentEnv` ; ou utiliser `gymnasium.make_vec_env` avec `SubprocVecEnv`

### Étape 2 — Modèle (Actor + Critic)

```
Observation (24,)
  → RunningNorm (mean/std courant)
  → Dense(256, tanh)  ←─── Critic path (séparé, plus large)
  → Dense(256, tanh)
  → Dense(1)          → V(s)
  ─────────────────────────────────────────
  → Dense(128, tanh)  ←─── Actor path (plus étroit)
  → Dense(128, tanh)
  → Dense(4)          → μ (tanh borné)
  → log_std (4,)     → tf.Variable trainable
```

**Points clés** :
- **Actor et Critic **complètement séparés** (pas de tronc commun)** — pour états basse dimension (24D), éviter l'interférence de gradient.actor et critic apprennent des représentations distinctes. ([[What Matters In On-Policy RL? — arXiv:2006.05990](https://arxiv.org/abs/2006.05990), [[Decoupled PPO — arXiv:2503.06343](https://arxiv.org/html/2503.06343v1)])
- **Critic plus large que Actor** — le critic a besoin de plus de capacité représentative ([[Honey, I Shrunk The Actor — arXiv:2102.11893](https://arxiv.org/pdf/2102.11893))

### Étape 3 — Agent PPO

Implémentation interne (pas de lib externe pour mieux comprendre) :

- ** rollout ** : collect `n_steps` transitions avec le policy courant
- ** clipped objective ** : `min(ratio * advantage, clip * advantage)` (PPO paper ε=0.2)
- ** value function loss with clipping ** : `value_clipped = V_old + clamp(V_new - V_old, -clip, +clip)` — [Engstrom et al.](https://arxiv.org/pdf/2005.12729.pdf)
- ** entropy bonus** : encourage l'exploration — fixer `ent_coef=0.001` (pas de schedule, pas 0.01 — trop d'exploration désordonnée)
- ** Adam optimizer** avec `ε=1e-5` et **linear LR annealing**
- ** Gradient clipping** par global norm (`max_grad_norm = 0.5`)
- ** Observation normalization** : running mean/std (màj par batch) — **sans clipping aprè normalize** ([[Andrychowicz et al. 2021](https://arxiv.org/pdf/2006.05990.pdf))
- ** Reward clipping** : `[-10, 10]`
- ** Advantage normalization** : normaliser les avantages avant l'objectif PG

Hyperparamètres principaux :

| Param | Valeur | Source |
|---|---|---|
| `n_envs` | 8–16 | [[sgoodfriend/ppo-BipedalWalker-v3](https://huggingface.co/sgoodfriend/ppo-BipedalWalker-v3)] |
| `n_steps` | 2048 | standard |
| `n_epochs` | 10 | standard |
| `batch_size` | 64 | standard |
| `clip_ratio` | 0.2 | PPO paper |
| `gamma` | 0.99 | standard |
| `lam` (GAE) | 0.95 | standard |
| `lr` | 3e-4 | standard |
| `lr_annealing` | linear 3e-4 → 0 | Engstrom |
| `value_coef` | 0.5 | standard |
| `entropy_coef` | **0.001** (fixe) | [[sgoodfriend/ppo-BipedalWalker-v3](https://huggingface.co/sgoodfriend/ppo-BipedalWalker-v3)] |
| `max_grad_norm` | 0.5 | Engstrom |
| `value_clip` | 0.2 | Engstrom |
| `rew_clip` | [-10, 10] | Engstrom |
| `obs_clip` | **drop (inutile post-norm)** | [[Andrychowicz et al. 2021](https://arxiv.org/pdf/2006.05990.pdf)] |

---

## Étape 4 — Entraînement

- **Multi-environments (8–16)** : utiliser `gymnasium.make_vec_env` avec `SubprocVecEnv` — stabilise les advantage estimates et accélère l'entraînement ([[sgoodfriend/ppo-BipedalWalker-v3](https://huggingface.co/sgoodfriend/ppo-BipedalWalker-v3)]). Si un seul env : doubler le nombre de steps/epochs.
- GAE (Generalized Advantage Estimation, λ=0.95, γ=0.99) pour les avantages
- Multi-epoch SGD (10 epochs par buffer de rollout)
- Logging TensorBoard : reward episode, loss totale, policy loss, value loss, entropy, KL divergence, learning rate
- Checkpointing du modèle toutes les N itérations (ModelCheckpoint TensorFlow)
- Early stopping si reward moyen > 300 sur 100 eps consécutifs
- Test eval (100 episodes) toutes les 50 itérations pour suivre la convergence
- **Multiple seeds (3–5)** : PPO est seed-sensitive sur BipedalWalker — ne pas conclure sur un seul run

---

## Étape 5 — Évaluation

- Rendu graphique avec `render_mode='human'`
- Collecte de stats : reward moyen, std, min, max sur N épisodes
- Comparaison avant/après entraînement

### Étape 6 — Améliorations optionnelles

- [ ] Hardcore terrain (piètres, escaliers)
- [ ] Curriculum learning (easy → hard)
- [ ] Wrapper `NormalizeObservation` / `NormalizeReward`
- [ ] HuggingFace Hub integration (upload du modèle)

---

## Fichiers à modifier

| Fichier | Description |
|---|---|
| `src/config.py` | Hyperparamètres |
| `src/model.py` | Architecture réseau |
| `src/agent.py` | Logique PPO |
| `src/training.py` | Boucle principale |
| `src/evaluation.py` | Évaluation |
| `scripts/train.py` | Entry point |
| `scripts/evaluate.py` | Entry point |
| `requirements.txt` | Dépendances |

---

## Tests / Validation

1. **Sanity check env** : `gym.make('BipedalWalker-v3')` + step aléatoire → verify shapes
2. **Forward pass** : une observation à travers le réseau → output (4,) + value (1,)
3. **Rollout test** : 1 itération de collecte → shapes OK, pas de NaN
4. **Training loop** : 100 étapes → loss décroissante, pas de NaN/Inf
5. **Convergence test** : runner complet → reward moyen > 300

---

## Risques et questions ouvertes

| Risque | Probabilité | Mitigation |
|---|---|---|
| **NaN/Inf dans le policy loss** | Haute | `tf.stop_gradient` sur les ratios, observation/reward clipping, gradient clipping |
| **Instabilité en fin d'entraînement** | Moyenne | LR annealing (linear decay à 0), value clipping, KL monitoring |
| **Temps d'entraînement > 10M steps** | Haute | **Multi-envs (8-16 parallèles) essentiels** — un seul env n'est pas assez stable ; config HF de référence utilise 10M steps total avec 16 envs [[sgoodfriend/ppo-BipedalWalker-v3](https://huggingface.co/sgoodfriend/ppo-BipedalWalker-v3)] |
| **Dead legs / collapse** | Basse | Entropy bonus, observation normalization, random seed fixe |
| **Box2D rendering crash** | Basse | headless si nécessaire (`render_mode=None`) |

### Questions ouvertes

1. **Stagger vs hardcore** : commencer sur stagger (défaut), porter à hardcore si temps le permet
2. **Separate vs shared trunk** : la littérature confirme maintenant que **séparé est meilleur** pour continuous control basse dimension — critic plus large que actor
3. **Nombre de layers** : 2–3 layers — le critic peut aller jusqu'à 3×256, actor plus étroit (2×128)
4. **TensorFlow vs JAX/Flax** : choix TensorFlow par confort, mais JAX serait ~2× plus rapide
5. **Reward scaling** : le clipping [-10, 10] suffit ; pas de reward shaping additionnel recommandé ([[LLM reward design paper](https://dl.acm.org/doi/full/10.1145/3778534.3778593)])

---

## Prochaines actions

1. ✅ Établir ce plan
2. ✅ Recherche (sources, hyperparamètres, improvements)
3. → Créer la structure de fichiers et `requirements.txt`
4. → Implémenter `config.py` et `model.py`
5. → Implémenter `agent.py` (PPO)
6. → Implémenter `training.py`
7. → Implémenter `scripts/train.py`
8. → Sanity checks (env, forward pass, rollout)
9. → Lancer l'entraînement
10. → Évaluation + rendu