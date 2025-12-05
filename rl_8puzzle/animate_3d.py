from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image

from rl_8puzzle.env import EightPuzzleEnv, GOAL_STATE, ACTIONS
from rl_8puzzle.train_q_learning import train, save_q
import imageio.v2 as imageio

State = Tuple[int, ...]


# ---------- Q-table helpers ----------


def load_or_train_q(path: str | Path = "rl_8puzzle/q_table.pkl"):
    import pickle

    path = Path(path)
    if path.exists():
        print(f"[animate] Loading existing Q-table from {path}")
        with path.open("rb") as f:
            return pickle.load(f)

    print("[animate] Q-table not found, training a smaller one for demo…")
    Q = train(
        num_episodes=20000,
        max_steps=80,
        scramble_moves=20,
    )
    save_q(Q, path)
    print(f"[animate] Saved Q-table to {path}")
    return Q


def greedy_trajectory(env: EightPuzzleEnv, Q, max_steps: int = 80) -> List[State]:
    """Generate a trajectory of states using the greedy policy from Q."""
    state = env.state
    path = [state]

    for _ in range(max_steps):
        qs = [Q.get((state, a), 0.0) for a in ACTIONS]
        best_idx = int(np.argmax(qs))
        action = ACTIONS[best_idx]

        state, _, done, _ = env.step(action)
        path.append(state)

        if done:
            break

    return path


# ---------- geometric helpers ----------


def state_to_xy_dict(state: State) -> Dict[int, Tuple[float, float]]:
    """
    Map a 3x3 board state to tile positions (x,y) on the grid.

    Returns: {tile_id: (x, y)}. Tile 0 (blank) is omitted.
    """
    pos: Dict[int, Tuple[float, float]] = {}
    for idx, tile in enumerate(state):
        if tile == 0:
            continue
        row, col = divmod(idx, 3)
        # Use (col, 2-row) so that the 'top' row is visually at the top.
        x, y = float(col), float(2 - row)
        pos[tile] = (x, y)
    return pos


def build_interpolated_frames(states: List[State], substeps: int = 8):
    """
    Given a list of discrete states [s0, s1, ..., sT], build a list of
    'frames'. Each frame is a list of (tile, x, y) positions, with
    smooth linear interpolation between moves.

    substeps: number of interpolation slices per move.
    """
    frames: List[List[Tuple[int, float, float]]] = []

    if not states:
        return frames

    # Start with exact initial positions
    pos0 = state_to_xy_dict(states[0])
    tiles = sorted(pos0.keys())
    initial_frame = [(tile, *pos0[tile]) for tile in tiles]
    frames.append(initial_frame)

    for i in range(len(states) - 1):
        pos_start = state_to_xy_dict(states[i])
        pos_end = state_to_xy_dict(states[i + 1])

        tiles = sorted(pos_start.keys())

        # substeps frames, excluding t=0 (already added), including t=1 at the end
        for k in range(1, substeps + 1):
            t = k / substeps
            frame: List[Tuple[int, float, float]] = []
            for tile in tiles:
                x0, y0 = pos_start[tile]
                x1, y1 = pos_end[tile]
                x = (1.0 - t) * x0 + t * x1
                y = (1.0 - t) * y0 + t * y1
                frame.append((tile, x, y))
            frames.append(frame)

    return frames


# ---------- 3D drawing helpers ----------


def setup_axes(ax: Axes3D):
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_zlim(0, 1.5)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_zticks([])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # High elevation so you see the tops clearly
    ax.view_init(elev=65, azim=-35)


from matplotlib import patheffects  # put this with your other imports at the top


def draw_frame(ax: Axes3D, frame: List[Tuple[int, float, float]]):
    # Clear axis and reset view
    ax.cla()
    setup_axes(ax)

    dx = dy = 0.9
    dz = 0.4

    # Distinct color per tile
    colors = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
    ]

    # Text outline style: white text with black stroke
    text_effects = [
        patheffects.Stroke(linewidth=3, foreground="black"),
        patheffects.Normal(),
    ]

    for tile, x, y in frame:
        color = colors[(tile - 1) % len(colors)]

        # Draw 3D cube
        ax.bar3d(
            x + 0.05,
            y + 0.05,
            0.0,
            dx,
            dy,
            dz,
            shade=True,
            alpha=0.95,
            color=color,
            edgecolor="black",
        )

        # BIG, high-contrast number clearly above the cube
        ax.text(
            x + 0.5,
            y + 0.5,
            dz + 0.35,  # float well above the cube
            str(tile),
            ha="center",
            va="center",
            fontsize=26,  # much larger font
            weight="bold",
            color="white",  # white text
            path_effects=text_effects,  # black outline
            zorder=10,
        )


# ---------- Manual GIF creation (no FuncAnimation) ----------


def animate_frames_to_mp4(
    frames: List[List[Tuple[int, float, float]]],
    save_path: str | Path = "rl_8puzzle/solution_3d.mp4",
    fps: int = 8,
):
    """
    Render each frame with Matplotlib and save as an MP4 video using imageio-ffmpeg.
    This is more robust than GIF on some Windows setups.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 6))
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    setup_axes(ax)
    ax.set_title("8-Puzzle RL Solution (3D Sliding Animation)")

    print(f"[animate] Saving MP4 video to {save_path} at {fps} fps …")
    writer = imageio.get_writer(save_path, fps=fps)

    try:
        for i, frame in enumerate(frames):
            draw_frame(ax, frame)
            ax.set_title(f"8-Puzzle RL Solution – Frame {i + 1}/{len(frames)}")

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())  # (h, w, 4)

            # imageio expects uint8 array
            writer.append_data(buf)

        print("[animate] MP4 saved.")
    finally:
        writer.close()
        plt.close(fig)


def main():
    # 1) load / train Q
    Q = load_or_train_q()

    # 2) repeatedly scramble until we get a decently long solution
    env = EightPuzzleEnv(scramble_moves=40)
    min_moves = 10  # require at least this many moves for a nice animation

    for attempt in range(20):
        start_state = env.reset()
        states = greedy_trajectory(env, Q)
        moves = len(states) - 1
        if moves >= min_moves:
            break

    print("[animate] Start state:", start_state)
    print("[animate] Goal state:", GOAL_STATE)
    print(f"[animate] Trajectory length (moves): {moves}")

    # 3) build smooth interpolated frames
    frames = build_interpolated_frames(states, substeps=10)
    print(f"[animate] Number of frames (with interpolation): {len(frames)}")

    # 4) render to MP4 (slower FPS so motion is obvious)
    animate_frames_to_mp4(
        frames,
        save_path="rl_8puzzle/solution_3d.mp4",
        fps=6,  # 6 frames per second
    )


if __name__ == "__main__":
    main()
