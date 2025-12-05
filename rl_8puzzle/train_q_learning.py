from __future__ import annotations

import random
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

from rl_8puzzle.env import EightPuzzleEnv, ACTIONS

State = Tuple[int, ...]
QKey = Tuple[State, int]
QTable = Dict[QKey, float]


def epsilon_greedy(Q: QTable, state: State, epsilon: float) -> int:
    """ε-greedy action selection."""
    if random.random() < epsilon:
        return random.choice(ACTIONS)

    qs = [Q[(state, a)] for a in ACTIONS]
    max_q = max(qs)
    best_actions = [a for a, q in zip(ACTIONS, qs) if q == max_q]
    return random.choice(best_actions)


def train(
    num_episodes: int = 50_000,
    max_steps: int = 100,
    gamma: float = 0.99,
    alpha: float = 0.1,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.01,
    scramble_moves: int = 30,
) -> QTable:
    """Tabular Q-learning for the 8-puzzle."""
    env = EightPuzzleEnv(scramble_moves=scramble_moves)
    Q: QTable = defaultdict(float)

    for episode in range(num_episodes):
        state = env.reset()

        # Linear ε decay
        frac = episode / max(num_episodes - 1, 1)
        epsilon = epsilon_start * (1.0 - frac) + epsilon_end * frac

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Q-learning update
            max_next = max(Q[(next_state, a)] for a in ACTIONS)
            old_value = Q[(state, action)]
            Q[(state, action)] = old_value + alpha * (
                reward + gamma * max_next - old_value
            )

            state = next_state
            if done:
                break

        if (episode + 1) % 5000 == 0:
            print(
                f"[train] Episode {episode + 1}/{num_episodes}, epsilon={epsilon:.4f}"
            )

    return Q


def save_q(Q: QTable, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        # dict() to remove defaultdict behaviour
        pickle.dump(dict(Q), f)


def main() -> None:
    print("[train] Starting Q-learning for 8-puzzle…")
    Q = train()
    save_q(Q, "rl_8puzzle/q_table.pkl")
    print("[train] Done. Saved Q-table → rl_8puzzle/q_table.pkl")


if __name__ == "__main__":
    main()
