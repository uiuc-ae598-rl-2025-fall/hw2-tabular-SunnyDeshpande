#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable

# ------------------------- Shared config -------------------------
GAMMA = 0.95
MAP_4x4 = ["SFFF",
           "FHFH",
           "FFFH",
           "HFFG"]
SEED = 42

A_LEFT, A_DOWN, A_RIGHT, A_UP = 0, 1, 2, 3
ARROW = {A_LEFT: '←', A_DOWN: '↓', A_RIGHT: '→', A_UP: '↑'}

# Evaluate every EVAL_EVERY steps using N_EVAL_EPISODES episodes
EVAL_EVERY = 1000
N_EVAL_EPISODES = 50
MAX_TRAIN_STEPS = 100_000
MAX_EP_LEN = 100


def build_env(is_slippery: bool):
    env = gym.make("FrozenLake-v1", desc=MAP_4x4, is_slippery=is_slippery)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    return env

def greedy_action(Q, s):
    return int(np.argmax(Q[s]))

def epsilon_greedy_action(Q, s, epsilon, num_A):
    if np.random.random() < epsilon:
        return np.random.randint(num_A)
    else:
        return greedy_action(Q,s)
    
def evaluate_policy(env, Q, num_episodes=50, max_len=100):
    total = 0.0
    for _ in range(num_episodes):
        s, _ = env.reset(seed=np.random.randint(0, 10**9))
        G = 0.0
        for _ in range(max_len):
            a = greedy_action(Q, s)
            s, r, terminate, truncate, _ = env.step(a)
            G += r
            if terminate or truncate:
                break
        total += G
    return total/num_episodes

def plot_eval_curves(eval_log: Dict[str, Tuple[List[int], List[float]]], title_suffix=""):
    plt.figure()
    for label, (steps, rets) in eval_log.items():
        plt.plot(steps, rets, label=label)
    plt.xlabel("Environment steps")
    plt.ylabel(f"Average return over {N_EVAL_EPISODES} eval episodes")
    plt.title(f"Evaluation Return vs Steps {title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_policy_and_value(Q: np.ndarray, nrow=4, ncol=4, title_prefix=""):
    V = Q.max(axis=1)
    policy = Q.argmax(axis=1)

    # Policy arrows
    plt.figure()
    for s in range(nrow*ncol):
        r, c = divmod(s, ncol)
        plt.text(c + 0.5, r + 0.5, ARROW[policy[s]], ha='center', va='center', fontsize=16)
    for r in range(nrow+1):
        plt.plot([0, ncol], [r, r], color='black', linewidth=0.5)
    for c in range(ncol+1):
        plt.plot([c, c], [0, nrow], color='black', linewidth=0.5)
    # annotate tiles
    for r in range(nrow):
        for c in range(ncol):
            plt.text(c + 0.05, r + 0.15, MAP_4x4[r][c], fontsize=9)
    plt.gca().invert_yaxis()
    plt.xticks([]); plt.yticks([])
    plt.title(f"{title_prefix} Policy (greedy wrt Q)")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Value heatmap
    grid = V.reshape(nrow, ncol)
    plt.figure()
    im = plt.imshow(grid, interpolation='nearest')
    for r in range(nrow):
        for c in range(ncol):
            plt.text(c, r, f"{grid[r,c]:.2f}", ha='center', va='center', fontsize=10)
    plt.colorbar(im)
    plt.xticks([]); plt.yticks([])
    plt.title(f"{title_prefix} State-Value V(s)=max_a Q(s,a)")
    plt.tight_layout()
    plt.show()

# Monte-Carlo First Visit Control (On-Policy)
def MC_First_Visit_Control(env: gym.Env,
                           gamma=GAMMA,
                           epsilon_start=1.0,
                           epsilon_end=0.05,
                           epsilon_decay_steps=40_000,
                           max_steps=MAX_TRAIN_STEPS,
                           max_ep_len=MAX_EP_LEN) -> Tuple[np.ndarray, Dict[str, Tuple[List[int], List[float]]]]:
    
    num_S = env.observation_space.n
    num_A = env.action_space.n
    Q = np.zeros((num_S, num_A), dtype=np.float64)
    returns_sum = [[0.0 for _ in range(num_A)] for _ in range(num_S)]
    returns_count = [[0 for _ in range(num_A)] for _ in range(num_S)]

    def current_epsilon(steps):
        if steps >= epsilon_decay_steps: 
            return epsilon_end
        else:
            frac = steps / float(epsilon_decay_steps)
            return epsilon_start + (epsilon_end - epsilon_start) * frac
    
    eval_steps, eval_returns = [], []
    steps = 0
    
    while steps < max_steps:
        episode = []
        s, _ = env.reset()
        for t in range(max_ep_len):
            eps = current_epsilon(steps)
            a = epsilon_greedy_action(Q, s, eps, num_A)
            s2, r, terminate, truncate, _ = env.step(a)
            episode.append((s, a, r))
            steps += 1
            s = s2
            if terminate or truncate or steps >= max_steps:
                break
            
            if steps % EVAL_EVERY == 0:
                avg_returns = evaluate_policy(env, Q, num_episodes=N_EVAL_EPISODES, max_len=max_ep_len)
                eval_steps.append(steps)
                eval_returns.append(avg_returns)
                
        visited = set()
        G = 0.0
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t = episode[t]
            G = gamma * G + r_t
            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                returns_sum[s_t][a_t] += G
                returns_count[s_t][a_t] += 1
                Q[s_t, a_t] = returns_sum[s_t][a_t] / max(1, returns_count[s_t][a_t])

    eval_log = {"MC (On-Policy)": (eval_steps, eval_returns)}
    return Q, eval_log

def SARSA_Control(env: gym.Env,
                  alpha=0.1,
                  gamma=GAMMA,
                  epsilon_start=1.0,
                  epsilon_end=0.05,
                  epsilon_decay_steps=60_000,
                  max_steps=MAX_TRAIN_STEPS,
                  max_ep_len=MAX_EP_LEN) -> Tuple[np.ndarray, Dict[str, Tuple[List[int], List[float]]]]:
    
    num_S = env.observation_space.n
    num_A = env.action_space.n
    Q = np.zeros((num_S, num_A), dtype=np.float64)
    
    def current_epsilon(steps):
        if steps >= epsilon_decay_steps:
            return epsilon_end
        frac = steps / float(epsilon_decay_steps)
        return epsilon_start + (epsilon_end - epsilon_start) * frac
    
    eval_steps, eval_returns = [], []
    steps = 0
    
    while steps < max_steps:
        s, _ = env.reset()
        a = epsilon_greedy_action(Q, s, current_epsilon(steps), num_A)
        for _ in range(max_ep_len):
            s2, r, terminate, truncate, _ = env.step(a)
            steps += 1 
            a2 = epsilon_greedy_action(Q, s2, current_epsilon(steps), num_A)
            td_target = r + gamma * Q[s2, a2] * (0 if (terminate or truncate) else 1)
            Q[s, a] += alpha * (td_target - Q[s, a])
            
            s, a = s2, a2
            if terminate or truncate or steps >= max_steps:
                break
            
            if steps % EVAL_EVERY == 0:
                avg_returns = evaluate_policy(env, Q, num_episodes=N_EVAL_EPISODES, max_len=max_ep_len)
                eval_steps.append(steps)
                eval_returns.append(avg_returns)

    eval_log = {"SARSA (On-Policy TD(0))": (eval_steps, eval_returns)}
    return Q, eval_log

def Q_Learning_Control(env: gym.Env,
                       alpha=0.1,
                       gamma=GAMMA,
                       epsilon_start=1.0,
                       epsilon_end=0.05,
                       epsilon_decay_steps=60_000,
                       max_steps=MAX_TRAIN_STEPS,
                       max_ep_len=MAX_EP_LEN) -> Tuple[np.ndarray, Dict[str, Tuple[List[int], List[float]]]]:
    num_S = env.observation_space.n
    num_A = env.action_space.n
    Q = np.zeros((num_S, num_A), dtype=np.float64)
    
    def current_epsilon(steps):
        if steps >= epsilon_decay_steps:
            return epsilon_end
        else:
            frac = steps / float(epsilon_decay_steps)
            return epsilon_start + (epsilon_end - epsilon_start) * frac
        
    eval_steps, eval_returns = [], []
    steps = 0
    
    while steps < max_steps:
        s, _ = env.reset()
        for _ in range(max_ep_len):
            a = epsilon_greedy_action(Q, s, current_epsilon(steps), num_A)
            s2, r, terminate, truncate, _ = env.step(a)
            steps += 1
            td_target = r + gamma * (0 if terminate or truncate else np.max(Q[s2]))
            Q[s, a] += alpha * (td_target - Q[s,a])
            
            s = s2
            if terminate or truncate or steps >= max_steps:
                break
            
            if steps % EVAL_EVERY == 0:
                avg_returns = evaluate_policy(env, Q, num_episodes=N_EVAL_EPISODES, max_len=max_ep_len)
                eval_steps.append(steps)
                eval_returns.append(avg_returns)
                
    eval_log = {"Q-Learning (Off-Policy TD(0))": (eval_steps, eval_returns)}
    return Q, eval_log



def run_all():
    np.random.seed(SEED)

    for slip in [True, False]:
        env = build_env(is_slippery=slip)
        slip_tag = "slippery=True" if slip else "slippery=False"

        # ---- MC ----
        Q_mc, mc_log = MC_First_Visit_Control(env)
        plot_eval_curves(mc_log, title_suffix=f"[MC, {slip_tag}]")
        plot_policy_and_value(Q_mc, title_prefix=f"MC – {slip_tag}")

        # ---- SARSA ----
        Q_sa, sa_log = SARSA_Control(env)
        plot_eval_curves(sa_log, title_suffix=f"[SARSA, {slip_tag}]")
        plot_policy_and_value(Q_sa, title_prefix=f"SARSA – {slip_tag}")

        # ---- Q-learning ----
        Q_ql, ql_log = Q_Learning_Control(env)
        plot_eval_curves(ql_log, title_suffix=f"[Q-learning, {slip_tag}]")
        plot_policy_and_value(Q_ql, title_prefix=f"Q-learning – {slip_tag}")

        env.close()


if __name__ == "__main__":
    run_all()