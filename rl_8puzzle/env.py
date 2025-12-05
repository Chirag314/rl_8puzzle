import random
from typing import Tuple, Dict, Any

GOAL_STATE: Tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    0,  # 0 = blank
)

# Actions: 0=up, 1=down, 2=left, 3=right
ACTIONS = (0, 1, 2, 3)


class EightPuzzleEnv:
    """
    Minimal 8-puzzle environment (3x3 sliding tiles) for RL.

    State: 9-tuple of ints 0..8 (0 = blank) in row-major order.
    Step reward: -1 for each move, +20 when reaching GOAL_STATE.
    Episodes start from a scrambled but solvable configuration.
    """

    def __init__(self, scramble_moves: int = 30) -> None:
        self.scramble_moves = scramble_moves
        self.state: Tuple[int, ...] = GOAL_STATE

    def _blank_pos(self, state: Tuple[int, ...]) -> Tuple[int, int]:
        idx = state.index(0)
        return divmod(idx, 3)  # (row, col)

    def _move(self, state: Tuple[int, ...], action: int) -> Tuple[int, ...]:
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

        # Invalid move → stay in place
        if not (0 <= r_new < 3 and 0 <= c_new < 3):
            return state

        new_state = list(state)
        old_idx = r * 3 + c
        new_idx = r_new * 3 + c_new
        new_state[old_idx], new_state[new_idx] = new_state[new_idx], new_state[old_idx]
        return tuple(new_state)

    def reset(self) -> Tuple[int, ...]:
        """Scramble from GOAL_STATE by random legal moves."""
        self.state = GOAL_STATE
        for _ in range(self.scramble_moves):
            action = random.choice(ACTIONS)
            self.state = self._move(self.state, action)
        return self.state

    def step(self, action: int):
        """
        Apply one action.

        Returns:
            next_state, reward, done, info
        """
        next_state = self._move(self.state, action)
        done = next_state == GOAL_STATE
        reward = 20.0 if done else -1.0
        self.state = next_state
        info: Dict[str, Any] = {}
        return next_state, reward, done, info

    def render(self) -> None:
        """Pretty-print the board to stdout."""
        s = self.state
        for row in range(3):
            print(s[row * 3 : (row + 1) * 3])
        print()
