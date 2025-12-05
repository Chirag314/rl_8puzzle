# RL 8-Puzzle (3x3 Sliding Tile)

This sub-project implements a tabular Q-learning agent for the 3×3 sliding tile
**8-puzzle** (numbers `1–8` with a blank). It is designed as a simple,
self-contained RL environment + agent that fits into your ML monorepo.

## Environment

- State: 9-tuple of integers `0..8` in row-major order, `0` = blank.
- Actions: `0=up`, `1=down`, `2=left`, `3=right`.
- Transition: deterministic swap of blank with neighbor; invalid moves leave the
  state unchanged.
- Reward:
  - `+20` when reaching the goal state `(1,2,3,4,5,6,7,8,0)`
  - `-1` otherwise (step cost)
- Episode:
  - Start from a scrambled but solvable configuration (random moves from goal)
  - Terminate on goal or after 100 steps

## Training

From the repo root:

```bash
python -m rl_8puzzle.train_q_learning
