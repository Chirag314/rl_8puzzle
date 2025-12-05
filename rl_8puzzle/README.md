# 🧩 RL 8-Puzzle — Q-Learning Agent + 3D Animated Solver

<p align="center">
  <video src="rl_8puzzle/media/example_solution_3d.mp4"
         controls loop muted playsinline width="480">
    Your browser does not support the video tag.
  </video>
</p>

A fully-functional **Reinforcement Learning project** that teaches an agent to solve the classic **8-Puzzle** sliding tile board using **Tabular Q-Learning** — and then visualizes the solution using a smooth **3D animated MP4 video**.

Every time you run the package, it can generate:

- A **brand-new solvable puzzle**,  
- A complete **solution trajectory** using the trained RL policy,  
- A high-quality **3D sliding animation**, saved as a video file.

---

# 🔖 Badges

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/rl-Q--learning-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-purple?style=flat-square" />
</p>

---

# 🌟 Features

### ✔ Reinforcement Learning agent (Tabular Q-Learning)  
### ✔ Automatically generates NEW puzzles each run  
### ✔ Full 3D animation of the solution  
### ✔ Command-line interface (CLI)  
### ✔ Clean modular code design  
### ✔ Automated tests (pytest)  
### ✔ Example MP4 embedded in README  
### ✔ Optional GIF generation  

---

# 📦 Project Structure

rl_8puzzle/
├─ main.py # Entry point (supports CLI arguments)
├─ env.py # 3×3 8-Puzzle environment
├─ n_puzzle_env.py # Optional N×N generalization
├─ train_q_learning.py # Q-learning implementation
├─ animate_3d.py # 3D animation engine (MP4 + GIF support)
├─ runner.py # "one command" generator for new puzzles + videos
├─ q_table.pkl # (generated) learned Q-table
├─ used_start_states.json # (generated) to avoid puzzle repetition
├─ media/
│ └─ example_solution_3d.mp4
└─ tests/
├─ test_env_8puzzle.py
└─ test_q_learning_8puzzle.py


---

# 🧠 RL Formulation

### **State**
A 9-tuple representing board configuration:

(1, 2, 3,
4, 5, 6,
7, 8, 0)


### **Action Space**
| Action | Effect |
|-------|--------|
| 0 | Move blank UP |
| 1 | Move blank DOWN |
| 2 | Move blank LEFT |
| 3 | Move blank RIGHT |

Invalid actions → state does **not** change.

### **Reward**
- Step: **–1**  
- Solved: **+20**

### **Episodes**
- Start from a scrambled configuration using valid moves from goal.
- Terminate on goal or step limit.

### **Q-Learning**

Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') – Q(s,a) ]


Policy:
- ε-greedy during training  
- Greedy during inference  

---

# 🚀 Quick Start

## Run the entire pipeline (new puzzle + animation)

```bash
python -m rl_8puzzle

### Generates:

rl_8puzzle/solution_3d.mp4

### If you run it again, it will try to produce another unique puzzle.
You can customize the puzzle difficulty, animation quality, and output:

python -m rl_8puzzle \
    --scramble 60 \
    --min-moves 15 \
    --fps 10 \
    --substeps 12 \
    --output rl_8puzzle/media/my_video.mp4

Available Flags

| Flag            | Meaning                                           |
| --------------- | ------------------------------------------------- |
| `--scramble N`  | Number of random moves when generating puzzle     |
| `--min-moves N` | Reject puzzles with solution shorter than N moves |
| `--fps N`       | Video frame rate                                  |
| `--substeps N`  | Interpolation frames between tile moves           |
| `--output PATH` | Save MP4 to custom location                       |
