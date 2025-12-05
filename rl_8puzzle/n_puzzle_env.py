import random
from typing import Tuple, Dict, Any, Iterable


class NPuzzleEnv:
    """
    General N x N sliding-tile puzzle for RL.

    Tiles: 1 .. (n*n - 1), 0 is blank.
    Goal state: (1, 2, 3, ..., n*n - 1, 0)
    Actions: 0=up, 1=down, 2=left, 3=right
    """

    ACTIONS = (0, 1, 2, 3)

    def __init__(self, size: int = 3, scramble_moves: int = 30):
        assert size >= 2
        self.size = size
        self.scramble_moves = scramble_moves

        self.goal_state: Tuple[int, ...] = tuple(list(range(1, size * size)) + [0])
        self.state: Tuple[int, ...] = self.goal_state

    # ---------- basic helpers ----------

    def _blank_pos(self, state: Tuple[int, ...]) -> Tuple[int, int]:
        idx = state.index(0)
        return divmod(idx, self.size)

    def _move(self, state: Tuple[int, ...], action: int) -> Tuple[int, ...]:
        n = self.size
        r, c = self._blank_pos(state)
        r_new, c_new = r, c

        if action == 0:  # up
            r_new -= 1
        elif action == 1:  # down
            r_new += 1
        elif action == 2:  # left
            c_new -= 1
        elif action == 3:  # right
            c_new += 1
        else:
            raise ValueError(f"Invalid action: {action}")

        if not (0 <= r_new < n and 0 <= c_new < n):
            return state  # invalid => no-op

        new_state = list(state)
        old_idx = r * n + c
        new_idx = r_new * n + c_new
        new_state[old_idx], new_state[new_idx] = new_state[new_idx], new_state[old_idx]
        return tuple(new_state)

    # ---------- RL API ----------

    def reset(self) -> Tuple[int, ...]:
        """Scramble goal state by random legal moves."""
        self.state = self.goal_state
        for _ in range(self.scramble_moves):
            a = random.choice(self.ACTIONS)
            self.state = self._move(self.state, a)
        return self.state

    def step(self, action: int):
        next_state = self._move(self.state, action)
        done = next_state == self.goal_state
        # same reward scheme for now
        reward = 20.0 if done else -1.0
        self.state = next_state
        info: Dict[str, Any] = {}
        return next_state, reward, done, info

    def render(self) -> None:
        n = self.size
        s = self.state
        for r in range(n):
            row = s[r * n : (r + 1) * n]
            print(" ".join("_" if x == 0 else str(x) for x in row))
        print()
