import imageio.v2 as imageio
from pathlib import Path

mp4_path = Path("rl_8puzzle/media/solution_3d.mp4")
gif_path = Path("rl_8puzzle/media/solution_3d.gif")

print(f"[make_gif] Reading {mp4_path} …")
reader = imageio.get_reader(mp4_path)

frames = []
for frame in reader:
    frames.append(frame)
reader.close()

print(f"[make_gif] Writing {gif_path} with {len(frames)} frames …")
imageio.mimsave(gif_path, frames, fps=10)
print("[make_gif] Done.")
