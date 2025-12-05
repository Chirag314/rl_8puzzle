from pathlib import Path
from PIL import Image


def main():
    gif_path = Path("rl_8puzzle/solution_3d.gif")
    if not gif_path.exists():
        print("GIF not found:", gif_path)
        return

    im = Image.open(gif_path)
    print("Format:", im.format)
    print("Size:", im.size)
    print("Is animated:", getattr(im, "is_animated", False))
    print("Number of frames:", getattr(im, "n_frames", 1))

    # Optionally export first few frames as PNGs to inspect
    out_dir = gif_path.parent / "frames_preview"
    out_dir.mkdir(exist_ok=True)
    max_frames = min(getattr(im, "n_frames", 1), 10)
    for i in range(max_frames):
        im.seek(i)
        im.save(out_dir / f"frame_{i:02d}.png")
    print(f"Saved first {max_frames} frames to {out_dir}")


if __name__ == "__main__":
    main()
