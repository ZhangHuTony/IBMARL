"""Convert an .mp4 video to a .gif, saved alongside the original file."""

import sys
from pathlib import Path

from moviepy import VideoFileClip
from PIL import Image
import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
VIDEO_PATH = Path("/home/connor/Desktop/Projects/IBMARL/results/rlfd_rl_comparison/ibmarl/ibmarl_navigation_2026-02-18_12-22-58/rendered_rollouts/render_logs/videos/rollout_seed_0_0.mp4")
FRAME_RATE = 8  # output GIF frames per second
COMPRESSION_QUALITY = 256  # max colours in the GIF palette (2–256)
RESIZE_FACTOR = 1.0  # scale factor for output dimensions (e.g. 0.5 = half size)
# -----------------------------------------------------------------------------


def convert(mp4_path: Path, fps: int, colors: int, resize: float) -> Path:
    clip = VideoFileClip(str(mp4_path))

    if resize != 1.0:
        clip = clip.resized(resize)

    frames = []
    for frame in clip.iter_frames(fps=fps, dtype="uint8"):
        img = Image.fromarray(frame)
        img = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        frames.append(img)
    clip.close()

    gif_path = mp4_path.with_suffix(".gif")
    duration_ms = int(1000 / fps)
    frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return gif_path


def main() -> None:
    if not VIDEO_PATH.exists():
        print(f"File not found: {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)
    if VIDEO_PATH.suffix.lower() != ".mp4":
        print(f"Expected an .mp4 file, got: {VIDEO_PATH.suffix}", file=sys.stderr)
        sys.exit(1)

    out = convert(VIDEO_PATH.resolve(), FRAME_RATE, COMPRESSION_QUALITY, RESIZE_FACTOR)
    print(f"Saved GIF to {out}")


if __name__ == "__main__":
    main()
