# BipedalWalker-v3 with PPO — TensorFlow GPU

Implementation from scratch of **Proximal Policy Optimization (PPO)** for the
`BipedalWalker-v3` environment (OpenAI Gymnasium / Box2D).  
Built with **TensorFlow/Keras**, optimized for **CUDA GPU** (mixed precision,
XLA JIT, `tf.data` pipeline), and heavily inspired by
*Engstrom et al. 2020* and *Schulman et al. 2017*.

---

## Table of contents

1. [What is PPO ?](#what-is-ppo)
2. [Architecture & design choices](#architecture--design-choices)
3. [GPU optimizations](#gpu-optimizations)
4. [Project structure](#project-structure)
5. [Installation](#installation)
6. [Quick start](#quick-start)
7. [Monitoring with TensorBoard](#monitoring-with-tensorboard)
8. [Evaluation](#evaluation)
9. [References](#references)

---

## What is PPO ?

PPO (**Proximal Policy Optimization**) is an *on-policy* reinforcement learning
algorithm that aims to be as simple to implement as vanilla policy gradients,
but as stable and sample-efficient as TRPO — without the expensive conjugate
gradient steps.

### 1. The core problem : policy updates are dangerous

In policy-gradient methods (REINFORCE, A2C…), you collect a batch of
experiences with your current policy `π_old`, then update the policy to maximize
the expected return.  The catch : **if you take too big a step, the new policy
`π` can collapse** (actions that were good become bad, or vice-versa).  You
would need a very small learning rate, making training slow.

PPO solves this with a **trust-region-like mechanism**, but using only first-order
gradient descent (no second-order methods).

### 2. The surrogate objective (clipped)

Instead of maximizing the raw probability of good actions, PPO maximizes a
*clipped* ratio between the new and old policy :

```
L^CLIP(θ) = E_t [ min( r_t(θ) * A_t ,  clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]
```

Where :

* `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)`  →  *probability ratio*
* `A_t`  →  *advantage* (how much better was this action than average ?)
* `ε`    →  *clip range* (usually 0.2)

**The trick :** if the ratio moves outside `[1-ε , 1+ε]`, the gradient is cut
— the policy is no longer encouraged to change further in that direction.  This
prevents *destructively large* updates while still allowing improvement.

### 3. Actor-Critic & GAE

PPO is almost always paired with two networks :

| Network | Output | Role |
|---------|--------|------|
| **Actor** | `μ(a\|s)` + `log_std` | Chooses actions |
| **Critic** | `V(s)` | Estimates how good a state is |

The **Critic** is used to compute the **Advantage** via **GAE** (Generalized
Advantage Estimation) :

```
Â_t = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + …
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

* `γ = 0.99` : discount factor
* `λ = 0.95` : GAE lambda (trade-off bias/variance)

This advantage tells the Actor : *"Was this action better than what the Critic
expected ?"*.

### 4. The complete loss

```
L_total = L^CLIP                (policy)
        + c_1 * L^VALUE         (value regression)
        + c_2 * H(π)            (entropy bonus)
```

* **Policy loss** : maximize clipped surrogate objective (minimize negative).
* **Value loss** : regression of `V(s)` toward GAE returns (with clipping,
  à la *Engstrom et al.*).
* **Entropy bonus** : force a little bit of exploration (prevents premature
  convergence).
* **Gradient clipping** : global norm capped at `0.5` for extra stability.

### 5. On-policy loop

```
for iteration:
    1. ROLLOUT   : collect N steps with π_old  → buffer
    2. GAE       : compute advantages & returns
    3. UPDATE    : run K epochs of minibatch SGD on the buffer
    4. DISCARD   : throw away the buffer (on-policy)
```

> **On-policy** means the data used for training must come from the *current*
> policy.  You cannot reuse old data (unlike SAC or DQN).

---

## Architecture & design choices

### Separated Actor & Critic

Following *Honey, I Shrunk The Actor* and *Engstrom et al. 2020*, the networks
are **completely separate** (no shared trunk) :

* **Actor** : 2 layers of 128 units (`tanh`, orthogonal init)
* **Critic** : 2 layers of 256 units (`tanh`, orthogonal init)

This avoids gradient interference and lets each network specialize.

### Action distribution

Continuous actions (4D torque) are modeled by an **independent Gaussian** per
dimension :

```
a ~ N(μ(s), diag(σ²))
μ(s) = tanh( actor(s) )   # bounded in [-1, 1]
σ    = exp(log_std)        # trainable parameter (not a network output)
```

`log_std` is a single trainable `tf.Variable` (shape `(4,)`), initialized to `0`
(σ = 1).  This is simpler and more stable than having the network output both
`μ` and `σ`.

### Observation normalization

A **RunningNorm** module maintains online mean & variance statistics (Welford
algorithm) over all observed states.  Observations are normalized before
entering the networks :

```python
obs_norm = (obs - running_mean) / sqrt(running_var + 1e-5)
```

This lives in NumPy for the CPU-side env loop, but is mirrored to
`tf.Variable`s for zero-copy GPU normalization.

### Hyperparameters

| Param | Value | Comment |
|-------|-------|---------|
| `n_envs` | 8 | SubprocVecEnv parallel rollouts |
| `n_steps` | 2048 | Steps per env before update (total batch = 8×2048 = 16384) |
| `n_epochs` | 10 | SGD passes over the batch |
| `batch_size` | 512 | Large minibatches for GPU saturation |
| `lr` | 3e-4 | Adam, linear decay to 0 |
| `clip_ratio` | 0.2 | PPO epsilon |
| `value_clip` | 0.2 | Critic value clipping |
| `γ` | 0.99 | Discount |
| `λ` | 0.95 | GAE lambda |
| `entropy_coef` | 0.01 | Exploration bonus |
| `max_grad_norm` | 0.5 | Global gradient clipping |

---

## GPU optimizations

This codebase is tuned for **NVIDIA GPUs** (tested on GTX 1060, RTX 3060/4090).

| Optimization | Effect |
|-------------|--------|
| **Mixed precision** (`float16`) | Forward pass & activations in `float16` → **2-3× throughput** on Tensor Cores. Weights stay `float32` for stability. |
| **XLA JIT** (`jit_compile=True`) | Compiles `_rollout_forward` and `_train_step` into fused CUDA kernels → less Python overhead, better GPU utilization. |
| **Fused forward** (`_rollout_forward`) | One graph call does `actor.sample + log_prob + critic.value` — single GPU kernel launch per env step. |
| **`tf.data` pipeline** | `shuffle → batch → prefetch(AUTOTUNE)` overlaps CPU data prep with GPU training. |
| **Zero-copy norm** | `RunningNorm` syncs stats to `tf.Variable`s so normalization stays on GPU (no CPU↔GPU transfer). |
| **Memory growth** | `tf.config.experimental.set_memory_growth` — avoids allocating all VRAM upfront. |

Enable/disable in `src/config.py` :

```python
use_mixed_precision = True   # float16 forward pass
xla_jit = True               # XLA compilation
gpu_memory_growth = True     # grow VRAM as needed
```

---

## Project structure

```
.
├── scripts/
│   ├── train.py          # Entry point : training loop
│   └── evaluate.py       # Entry point : evaluation / render
├── src/
│   ├── config.py         # PPOConfig dataclass (all hyperparams)
│   ├── model.py          # ActorNetwork, CriticNetwork, PPOModel, RunningNorm
│   ├── agent.py          # RolloutBuffer, PPOAgent (GAE + PPO update)
│   ├── training.py       # make_vec_env, TrainingRunner, TensorBoardLogger
│   └── evaluation.py     # evaluate_agent, load_and_evaluate
├── tests/
│   └── test_sanity.py    # Forward pass, rollout buffer, mini training loop
├── requirements.txt
└── README.md
```

---

## Installation

Requires **Python 3.10+**, a GPU with **CUDA 11.8+ / cuDNN 8+** is strongly
recommended (CPU training is ~20× slower).

```bash
# 1. Clone
cd bipedal-walker-tensorflow

# 2. Install dependencies (uv or pip)
uv pip install -r requirements.txt
# or
pip install -r requirements.txt
```

Dependencies : `tensorflow>=2.15`, `gymnasium[box2d]`, `stable-baselines3`,
`tensorboard`, `numpy`, `matplotlib`.

---

## Quick start

### Training

```bash
# Default : 1 seed, 8 envs, 10M steps
uv run scripts/train.py

# Multiple seeds
uv run scripts/train.py --seeds 3

# Fewer steps for a quick test
uv run scripts/train.py --total-steps 500_000

# Custom LR / envs
uv run scripts/train.py --n-envs 16 --lr 1e-4
```

Outputs :
* Checkpoints every 50 iterations (`ckpt_iter<N>`)
* Final model (`final_model_*.{h5,npz}`)
* TensorBoard logs in `runs/ppo_BipedalWalker-v3_s0_<timestamp>/`

### Monitoring with TensorBoard

During (or after) training, launch TensorBoard in a **second terminal** :

```bash
tensorboard --logdir=runs/
```

Then open `http://localhost:6006`.

Tracked metrics :
* `rollout/ep_rew_mean` — mean episode reward
* `train/policy_loss`, `train/value_loss`, `train/entropy`
* `train/approx_kl` — KL divergence between old & new policy
* `train/clip_fraction` — % of samples where ratio is clipped
* `train/explained_variance` — how well the critic predicts returns (1.0 = perfect)
* `opt/learning_rate` — LR annealing curve
* `timing/fps` — training throughput

### Evaluation

```bash
# 100 episodes, deterministic (recommended for reporting)
uv run scripts/evaluate.py \
    --checkpoint runs/ppo_BipedalWalker-v3_s0_20260512_213622/final_model \
    --episodes 100

# Visualize 5 episodes (requires display / X11)
uv run scripts/evaluate.py \
    --checkpoint runs/ppo_BipedalWalker-v3_s0_20260512_213622/final_model \
    --episodes 5 --render

# Stochastic policy (for debugging exploration)
uv run scripts/evaluate.py \
    --checkpoint runs/ppo_BipedalWalker-v3_s0_20260512_213622/final_model \
    --episodes 100 --stochastic
```

> **Important** : pass the **prefix** only (`final_model`), not the full
> filename. The script auto-appends `_actor.weights.h5`, `_critic.weights.h5`,
> and `_obs_norm.npz`.

---

## References

1. **Schulman et al., 2017** — *Proximal Policy Optimization Algorithms*
   (arXiv:1707.06347)
2. **Engstrom et al., 2020** — *Implementation Matters in Deep RL: A Case Study
   on PPO and TRPO* (arXiv:2005.12729)
3. **Ilyas et al., 2020** — *A Closer Look at Deep Policy Gradients*
   (arXiv:2009.10897) — *Honey, I Shrunk The Actor*
4. **OpenAI Baselines** — `ppo2` implementation
5. **Stable-Baselines3** — `PPO` reference hyperparameters

---

## License

MIT — do whatever you want, but keep the references if you publish.

> *“If you understand PPO, you understand 80% of modern on-policy RL.”*
