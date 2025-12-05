from collections import defaultdict

from rl_8puzzle.env import EightPuzzleEnv, ACTIONS
from rl_8puzzle.train_q_learning import train


def greedy_action(Q, state):
    """Pick greedy action given Q-table."""
    qs = [Q.get((state, a), 0.0) for a in ACTIONS]
    max_idx = max(range(len(ACTIONS)), key=lambda i: qs[i])
    return ACTIONS[max_idx]


def test_train_returns_non_empty_q_table():
    Q = train(
        num_episodes=1000,  # keep tests reasonably fast
        max_steps=80,
        scramble_moves=10,
    )
    assert isinstance(Q, dict)
    assert len(Q) > 0


def test_trained_policy_solves_simple_states():
    """
    Weak behavioral test:
    after some training, the greedy policy should be able
    to solve at least some scrambled states within a small
    move budget.
    """
    Q = train(
        num_episodes=2000,
        max_steps=80,
        scramble_moves=10,
    )

    env = EightPuzzleEnv(scramble_moves=5)
    successes = 0
    trials = 10

    for _ in range(trials):
        state = env.reset()

        for _ in range(40):  # max moves per trial
            action = greedy_action(Q, state)
            state, _, done, _ = env.step(action)
            if done:
                successes += 1
                break

    # We don't demand perfection, just that it solves *something*
    assert successes >= 1
