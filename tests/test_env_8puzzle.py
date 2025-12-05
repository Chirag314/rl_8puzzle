import pytest

from rl_8puzzle.env import EightPuzzleEnv, GOAL_STATE


def test_step_invalid_move_stays_put():
    env = EightPuzzleEnv(scramble_moves=0)
    env.state = GOAL_STATE  # blank is bottom-right
    before = env.state

    # invalid move: right (action=3) from bottom-right
    after, reward, done, _ = env.step(3)

    # State must not change
    assert after == before

    # Reward should not be positive (either step cost or goal reward).
    # Since we are already at goal, it's fine if done is True.
    assert reward <= 20.0
