from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

from manim import Scene, Square, Text, VGroup, DOWN, UP

from rl_8puzzle.env import EightPuzzleEnv, ACTIONS
from rl_8puzzle.train_q_learning import load_q  # we will create this helper


State = Tuple[int, ...]


def greedy_trajectory(env: EightPuzzleEnv, Q, max_steps: int = 80) -> List[State]:
    state = env.state
    path = [state]
    for _ in range(max_steps):
        qs = [Q.get((state, a), 0.0) for a in ACTIONS]
        best_idx = max(range(len(ACTIONS)), key=lambda i: qs[i])
        action = ACTIONS[best_idx]
        state, _, done, _ = env.step(action)
        path.append(state)
        if done:
            break
    return path


class PuzzleManim(Scene):
    def construct(self):
        # Load Q-table
        from rl_8puzzle.animate_3d import load_or_train_q

        Q = load_or_train_q()  # reuse helper

        env = EightPuzzleEnv(scramble_moves=20)
        env.reset()
        states = greedy_trajectory(env, Q)

        # Create 3x3 grid of squares
        tiles = {}
        group = VGroup()
        size = 1.0

        def pos_for_index(idx):
            row, col = divmod(idx, 3)
            # y: invert so row 0 is at top
            return (col - 1) * size * 1.1 * RIGHT + (1 - row) * size * 1.1 * UP

        from manim import RIGHT

        # First state
        s0 = states[0]
        for i, tile in enumerate(s0):
            if tile == 0:
                continue
            square = Square(side_length=size)
            square.move_to(pos_for_index(i))
            label = Text(str(tile), font_size=36)
            label.move_to(square.get_center())
            tile_group = VGroup(square, label)
            tiles[tile] = tile_group
            group.add(tile_group)

        self.play(group.fade_in())
        self.wait(0.5)

        # Animate each move
        for prev, nxt in zip(states[:-1], states[1:]):
            # find which tile moved by comparing positions
            prev_pos = {tile: i for i, tile in enumerate(prev) if tile != 0}
            next_pos = {tile: i for i, tile in enumerate(nxt) if tile != 0}

            # the moved tile changed index
            moved_tile = None
            for tile in prev_pos:
                if prev_pos[tile] != next_pos[tile]:
                    moved_tile = tile
                    break

            if moved_tile is None:
                continue

            new_idx = next_pos[moved_tile]
            target_pos = pos_for_index(new_idx)
            self.play(tiles[moved_tile].animate.move_to(target_pos), run_time=0.3)

        self.wait(1)
