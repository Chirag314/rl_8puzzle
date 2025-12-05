from __future__ import annotations

import json
from pathlib import Path
from typing import Set, Tuple

from rl_8puzzle.env import EightPuzzleEnv, GOAL_STATE
from rl_8puzzle.animate_3d import (
    load_or_train_q,
    greedy_trajectory,
    build_interpolated_frames,
    animate_frames_to_mp4,
)

State = Tuple[int, ...]
HISTORY_PATH = Path(__file__).with_name("used_start_states.json")


def _load_history() -> Set[State]:
    if not HISTORY_PATH.exists():
        return set()
    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # stored as list of lists -> convert back to tuples
    return {tuple(s) for s in data}


def _save_history(history: Set[State]) -> None:
    # store as list of lists for JSON
    data = [list(s) for s in history]
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main(
    scramble_moves: int = 40,
    min_moves: int = 10,
    substeps: int = 10,
    fps: int = 6,
    video_path: str | Path = "rl_8puzzle/solution_3d.mp4",
) -> None:
    """
    One-shot runner:

    - Loads (or trains) the Q-table.
    - Repeatedly scrambles until it finds:
        * a start state that hasn't been used before, AND
        * a greedy solution with at least `min_moves` moves.
    - Saves that start state to history so it's not reused.
    - Builds smooth interpolated frames.
    - Renders a 3D MP4 animation.
    """
    Q = load_or_train_q()

    env = EightPuzzleEnv(scramble_moves=scramble_moves)
    used_starts = _load_history()

    chosen_state: State | None = None
    chosen_states_traj: list[State] | None = None

    max_attempts = 200

    for attempt in range(max_attempts):
        start_state = env.reset()
        if start_state in used_starts:
            continue

        states = greedy_trajectory(env, Q)
        moves = len(states) - 1

        if moves < min_moves:
            # too trivial; try again
            continue

        chosen_state = start_state
        chosen_states_traj = states
        print(
            f"[runner] Selected start state on attempt {attempt + 1}: "
            f"{start_state} (moves to solve: {moves})"
        )
        break

    if chosen_state is None or chosen_states_traj is None:
        # fallback: just use the last sampled scramble
        print(
            "[runner] Could not find a new + sufficiently long puzzle "
            "within attempt limit. Using last scramble."
        )
        chosen_state = start_state
        chosen_states_traj = states

    used_starts.add(chosen_state)
    _save_history(used_starts)

    print("[runner] Start state:", chosen_state)
    print("[runner] Goal state:", GOAL_STATE)
    print(f"[runner] Greedy trajectory length (moves): {len(chosen_states_traj) - 1}")

    # Build smooth frames and render video
    frames = build_interpolated_frames(chosen_states_traj, substeps=substeps)
    print(f"[runner] Frames with interpolation: {len(frames)}")

    animate_frames_to_mp4(
        frames,
        save_path=video_path,
        fps=fps,
    )
    print(f"[runner] Done. Video saved to {video_path}")
