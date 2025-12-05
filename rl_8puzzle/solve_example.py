from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

from rl_8puzzle.env import EightPuzzleEnv, ACTIONS

State = Tuple[int, ...]
QKey = Tuple[State, int]
QTable = Dict[QKey, float]


def load_q(path: str | Path) -> QTable:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def greedy_solve(env: EightPuzzleEnv, Q: QTable, max_steps: int = 100):
    """Follow the greedy policy from the current env.state."""
    state = env.state
    path = [state]

    for _ in range(max_steps):
        qs = [Q.get((state, a), 0.0) for a in ACTIONS]
        best_idx = max(range(len(ACTIONS)), key=lambda i: qs[i])
        best_action = ACTIONS[best_idx]

        state, _, done, _ = env.step(best_action)
        path.append(state)

        if done:
            break

    return path


def print_board(state: State) -> None:
    for r in range(3):
        row = state[r * 3 : (r + 1) * 3]
        print(" ".join("_" if x == 0 else str(x) for x in row))
    print()


def main() -> None:
    Q = load_q("rl_8puzzle/q_table.pkl")
    env = EightPuzzleEnv(scramble_moves=20)
    start_state = env.reset()

    print("Start state:")
    print_board(start_state)

    path = greedy_solve(env, Q)

    print(f"Solved in {len(path) - 1} moves.")
    print("Trajectory:")
    for s in path:
        print_board(s)


if __name__ == "__main__":
    main()
