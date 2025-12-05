from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from rl_8puzzle.env import EightPuzzleEnv, ACTIONS
from rl_8puzzle.animate_3d import load_or_train_q, greedy_trajectory

State = Tuple[int, ...]


def export_trajectory(path: str | Path = "rl_8puzzle/trajectory.json"):
    Q = load_or_train_q()
    env = EightPuzzleEnv(scramble_moves=20)
    env.reset()
    states: List[State] = greedy_trajectory(env, Q)

    data = {
        "size": 3,
        "states": [list(s) for s in states],  # list of length-9 lists
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[export] Saved trajectory with {len(states) - 1} moves to {path}")


if __name__ == "__main__":
    export_trajectory()
