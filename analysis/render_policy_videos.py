"""
Render videos of trained policies from finished runs, for the poster.

    python -m analysis.render_policy_videos --tag buzzwire3 --variant rlfd \\
        --seed median --episodes 3 --label "RLfD" --out poster/videos/buzzwire_rlfd.mp4
    python -m analysis.render_policy_videos --tag paper5 --variant bc_eval --out nav_teacher.mp4
    python -m analysis.render_policy_videos --batch poster            # the whole poster set
    python -m analysis.render_policy_videos --batch poster --dry-run  # seeds + paths only

A run's ``config.yaml`` rebuilds the environment it trained in (legacy or
binary schema, horizon, scenario) and ``checkpoints/policy_checkpoint.pt``
(``{group: actor state_dict}``, saved by every learner at the end of a run)
its actor -- MADDPG, RLfD, RFT and IBMARL share one actor architecture
(src/experiments/ibmarl/networks.py::build_rl_policies), so one builder covers
all four.  ``bc_eval*`` rows render the frozen R2BC teacher instead.  IBMARL
rows are the RL actors alone (the decentralised policy the poster videos are
about); the arbitrated RL+IL policy is not rendered.

Each video is N deterministic episodes (no exploration noise, one episode per
env seed ``seed + 20000 + i``, the offset the training-time recorder uses), a
short hold on each episode's last frame so successes do not read as cuts, a
burned-in label, and a matching GIF and still.  Rendering needs a display
(VMAS draws with pyglet) and PyAV for the mp4 encode; both are on the
workstation, neither on the cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
RENDER_SEED_OFFSET = 20_000          # base_marl_experiment._record_policy_video

# Viewer settings applied to the VMAS scenario before its first render.  The
# buzz-wire corridor is tall and narrow: the 700x700 default is ~98% white.
VIEW = {
    # zoom 0.9 shows the whole corridor plus the agents' starting positions
    # outside it with a margin (0.75, paper_figures_two_task.py's crop for the
    # paper's task panel, clips the agents at the top edge); the viewer is
    # sized up so the auto-cropped corridor keeps ~200 px of width.
    "buzz_wire": dict(viewer_size=(600, 1170), viewer_zoom=0.9),
}
TASK_TITLE = {"navigation": "Navigation", "buzzwire": "Buzz-wire", "transport": "Transport"}
# R2BC teacher runs whose checkpoints are bundled in teachers/ (md5-identical),
# so a frozen config's dead absolute path can be re-rooted.
BUNDLED_TEACHERS = {
    "navigation_r2bc_decent_20260129_201300": "teachers/navigation",
    "buzzwire_r2bc_decent_20260826_225653": "teachers/buzz_wire_12demo",
    "buzzwire_r2bc_decent_20260921_021440": "teachers/buzz_wire",
    "transport_r2bc_decent_20260821_154402": "teachers/transport",
}


@dataclass(frozen=True)
class VideoRow:
    task: str          # navigation | buzzwire | transport (file-name key)
    tag: str           # results/<tag>
    variant: str
    label: str         # burned into the frames
    slug: str          # poster/videos/<task>_<slug>.mp4
    seed: int | str = "median"


def poster_rows(nav_tag: str, bw_tag: str, tr_tag: str) -> list[VideoRow]:
    """The poster set: six rows per task plus the buzz-wire human-teacher arm."""
    common = [
        ("maddpg", "MADDPG", "maddpg"),
        ("rlfd", "RLfD", "rlfd"),
        ("rft", "RFT", "rft"),
        ("bc_eval", "Teacher (R2BC)", "teacher"),
        ("ibmarl_gated_a0.4", "IBMARL (RL actors)", "ibmarl"),
        ("ibmarl", "IBMARL w/o gated term (RL actors)", "ibmarl_nogate"),
    ]
    rows = []
    for task, tag in (("navigation", nav_tag), ("buzzwire", bw_tag), ("transport", tr_tag)):
        for variant, label, slug in common:
            rows.append(VideoRow(task, tag, variant, label, slug))
    for variant, label, slug in [
        ("bc_eval_human", "Human teacher (R2BC)", "teacher_human"),
        ("ibmarl_gated_a0.4_human", "IBMARL, human teacher (RL actors)", "ibmarl_human"),
        ("rft_human", "RFT, human demos", "rft_human"),
        ("rlfd_human", "RLfD, human demos", "rlfd_human"),
    ]:
        rows.append(VideoRow("buzzwire", bw_tag, variant, label, slug))
    return rows


# ----------------------------------------------------------------------------
# run discovery and seed choice
# ----------------------------------------------------------------------------
def run_dir(tag: str, variant: str, seed: int) -> Path:
    return ROOT / "results" / tag / variant / f"seed_{seed}"


def completed_seeds(tag: str, variant: str) -> list[int]:
    seeds = []
    for d in sorted((ROOT / "results" / tag / variant).glob("seed_*")):
        try:
            if json.load(open(d / "status.json")).get("ok"):
                seeds.append(int(d.name.split("_")[1]))
        except (OSError, ValueError):
            continue
    return sorted(seeds)


def final10(run: Path, protocol: str = "rl") -> float:
    """Mean of the last 10 evaluations under *protocol* (paper3_figures' rule)."""
    import pandas as pd
    from analysis import paper3_figures as pf
    df = pd.read_csv(run / "data" / "metrics.csv")
    col = pf.resolve_col(df, "eval_reward_mean", protocol)
    return float(df[col].dropna().tail(10).mean())


def pick_seed(tag: str, variant: str, seed: int | str, protocol: str = "rl") -> tuple[int, str]:
    """
    The seed to render.  ``median``: among finished seeds, the one whose
    final-10 evaluation is the median (lower median for an even count) -- a
    representative run rather than the best-looking one.  bc_eval rows have
    no checkpoint and the teacher is seed-independent, so seed 0.
    """
    if isinstance(seed, int) or (isinstance(seed, str) and seed.lstrip("-").isdigit()):
        return int(seed), "seed given explicitly"
    if variant.startswith("bc_eval"):
        return 0, "teacher is seed-independent"
    seeds = completed_seeds(tag, variant)
    if not seeds:
        raise FileNotFoundError(f"{tag}/{variant}: no finished seeds")
    finals = {s: final10(run_dir(tag, variant, s), protocol) for s in seeds}
    order = sorted(seeds, key=lambda s: finals[s])
    chosen = order[(len(order) - 1) // 2]
    detail = ", ".join(f"seed {s}: {finals[s]:.2f}" for s in order)
    return chosen, f"median final-10 ({protocol} protocol) of {len(order)} seeds [{detail}]"


def load_cfg(run: Path) -> dict:
    with open(run / "config.yaml") as f:
        return yaml.safe_load(f)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_teacher_checkpoint(cfg: dict) -> tuple[Path, str]:
    """
    The teacher checkpoint a run's config names, or -- when that absolute path
    belongs to a machine or directory layout that no longer exists (paper3's
    pre-move R2BC path) -- the same R2BC run re-rooted on ../R2BC/results or
    its bundled copy under teachers/.  The second value says which.
    """
    from src.util.paths import resolve_path
    raw = cfg.get("r2bc_checkpoint_path")
    if not raw:
        raise KeyError("config has no r2bc_checkpoint_path")
    p = resolve_path(raw)
    if p.is_file():
        return p, "as configured"
    run_name = Path(raw).parent.name
    candidates = [ROOT.parent / "R2BC" / "results" / run_name / "policy_checkpoint.pth"]
    if run_name in BUNDLED_TEACHERS:
        candidates.append(ROOT / BUNDLED_TEACHERS[run_name] / "policy_checkpoint.pth")
    for c in candidates:
        if c.is_file():
            return c, f"re-rooted from dead path {raw}"
    raise FileNotFoundError(f"teacher checkpoint not found: {raw} (tried {candidates})")


# ----------------------------------------------------------------------------
# environment and policies
# ----------------------------------------------------------------------------
def build_render_env(cfg: dict, device, seed: int):
    """
    One-sub-env copy of the run's environment with a pixel transform.  One
    sub-env because PixelRenderTransform draws sub-env 0 only and rollout
    stops at the first done -- with several sub-envs the recording would end
    when the FASTEST one finished while the one on screen is mid-episode.
    """
    from torchrl.record import PixelRenderTransform
    from src.environment.make_env import make_env

    c = dict(cfg)
    c["seed"] = int(seed)
    c["frames_per_batch"] = int(cfg.get("horizon", 100))   # -> exactly one sub-env
    c["render"] = False
    env = make_env(c, device)
    view = VIEW.get(cfg.get("scenario_name"), {})
    if view:
        scenario = env.base_env._env.scenario     # vmas.simulator.environment.Environment
        for k, v in view.items():
            setattr(scenario, k, v)
    env = env.append_transform(
        PixelRenderTransform(
            out_keys=["pixels"],
            preproc=lambda x: x.copy(),
            as_non_tensor=True,
            mode="rgb_array",
            visualize_when_rgb=False,
        )
    )
    return env


def build_actor_policy(cfg: dict, run: Path, env, device):
    from tensordict.nn import TensorDictSequential
    from src.experiments.ibmarl.networks import build_rl_policies

    ckpt = run / "checkpoints" / "policy_checkpoint.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"{run}: no checkpoints/policy_checkpoint.pt")
    policies = build_rl_policies(cfg, env, device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    for group, sd in state.items():
        if group not in policies:
            raise KeyError(f"{ckpt}: group {group!r} not in env groups {list(policies)}")
        policies[group].load_state_dict(sd, strict=True)
    policy = TensorDictSequential(*policies.values())
    policy.eval()
    return policy, ckpt


def build_teacher_policy(cfg: dict, env, device):
    from src.experiments.bc_eval_experiment import build_teacher_td_policy
    from src.experiments.ibmarl.networks import R2bcPolicy

    path, how = resolve_teacher_checkpoint(cfg)
    bc = R2bcPolicy(path, env, device)
    policy = build_teacher_td_policy(bc, env.group_map)
    policy.eval()
    return policy, path, how


# ----------------------------------------------------------------------------
# rollouts and frames
# ----------------------------------------------------------------------------
@dataclass
class Episode:
    env_seed: int
    length: int
    frames: list = field(repr=False)
    returns: dict
    terminated: bool
    truncated: bool | None
    goal_step: int | None


def _frames_from_rollout(out) -> list[np.ndarray]:
    """Reset frame + one frame per step, from the NonTensorStack of pixels."""
    first = out.get("pixels").tolist()[0][0]
    rest = out.get(("next", "pixels")).tolist()[0]
    frames = [np.asarray(first)] + [np.asarray(f) for f in rest]
    return [f.astype(np.uint8) for f in frames]


def rollout_episode(env, policy, env_seed: int, horizon: int, sparse: bool) -> Episode:
    from torchrl.envs import ExplorationType, set_exploration_type

    env.set_seed(int(env_seed))
    with torch.no_grad():
        with set_exploration_type(ExplorationType.MODE):
            out = env.rollout(horizon, policy=policy, break_when_any_done=True)
    T = int(out.batch_size[1])
    frames = _frames_from_rollout(out)
    returns = {}
    goal_step = None
    for group in env.group_map:
        ep = out.get(("next", group, "episode_reward"))[0, -1]
        returns[group] = float(ep.mean().item())
        if sparse:
            r = out.get(("next", group, "reward"))[0].reshape(T, -1).mean(-1)
            hit = torch.nonzero(r > -0.5, as_tuple=True)[0]
            if len(hit) and goal_step is None:
                goal_step = int(hit[0].item())
    term = out.get(("next", "terminated"))[0, -1]
    trunc = out.get(("next", "truncated"), None)
    return Episode(
        env_seed=int(env_seed), length=T, frames=frames, returns=returns,
        terminated=bool(term.any().item()),
        truncated=(bool(trunc[0, -1].any().item()) if trunc is not None else None),
        goal_step=goal_step,
    )


def auto_crop_box(frames: list[np.ndarray], pad: float = 0.06, white: int = 245,
                  stride: int = 5) -> tuple[int, int, int, int]:
    """Union bounding box of non-white pixels over the frames (sampled every
    *stride*), padded, rounded to even sizes (libx264 rejects odd ones)."""
    H, W = frames[0].shape[:2]
    y0, y1, x0, x1 = H, 0, W, 0
    for f in frames[::stride] + [frames[-1]]:
        mask = f.min(axis=-1) < white
        ys, xs = np.nonzero(mask)
        if len(ys):
            y0, y1 = min(y0, ys.min()), max(y1, ys.max())
            x0, x1 = min(x0, xs.min()), max(x1, xs.max())
    if y1 <= y0 or x1 <= x0:
        return 0, H, 0, W
    py, px = int(pad * (y1 - y0)), int(pad * (x1 - x0))
    y0, y1 = max(0, y0 - py), min(H, y1 + py + 1)
    x0, x1 = max(0, x0 - px), min(W, x1 + px + 1)
    if (y1 - y0) % 2:
        y1 = y1 - 1 if y1 - 1 > y0 else y1
    if (x1 - x0) % 2:
        x1 = x1 - 1 if x1 - 1 > x0 else x1
    return y0, y1, x0, x1


def _font(size: int):
    from PIL import ImageFont
    import matplotlib.font_manager as fm
    try:
        return ImageFont.truetype(fm.findfont("DejaVu Sans:bold"), size)
    except Exception:
        return ImageFont.load_default()


def burn_label(frame: np.ndarray, text: str, corner_text: str | None, font) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    m = max(4, img.width // 60)
    box = draw.textbbox((m, m), text, font=font)
    draw.rectangle((box[0] - m // 2, box[1] - m // 2, box[2] + m // 2, box[3] + m // 2),
                   fill=(255, 255, 255, 210))
    draw.text((m, m), text, font=font, fill=(20, 20, 20, 255))
    if corner_text:
        cb = draw.textbbox((0, 0), corner_text, font=font)
        w = cb[2] - cb[0]
        x = img.width - w - m
        box = draw.textbbox((x, m), corner_text, font=font)
        draw.rectangle((box[0] - m // 2, box[1] - m // 2, box[2] + m // 2, box[3] + m // 2),
                       fill=(255, 255, 255, 210))
        draw.text((x, m), corner_text, font=font, fill=(20, 20, 20, 255))
    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))


def fit_label(crop, frame_shape, label: str, n_episodes: int, min_font: int = 12):
    """
    Widen the crop box (symmetrically, within the frame) until the burned-in
    label and the episode counter fit side by side; returns (crop, font).
    A tall, narrow crop such as the buzz-wire corridor is otherwise narrower
    than its own caption.
    """
    from PIL import ImageDraw, Image
    y0, y1, x0, x1 = crop
    H, W = frame_shape[:2]
    font = _font(max(min_font, (x1 - x0) // 22))
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    m = max(4, (x1 - x0) // 60)
    need = draw.textbbox((0, 0), label, font=font)[2] + 3 * m
    if n_episodes > 1:
        need += draw.textbbox((0, 0), f"episode {n_episodes}/{n_episodes}", font=font)[2] + 3 * m
    if need > (x1 - x0):
        extra = need - (x1 - x0)
        x0 = max(0, x0 - extra // 2)
        x1 = min(W, x0 + max(need, x1 - x0))
        x0 = max(0, x1 - max(need, x1 - x0))
        if (x1 - x0) % 2:
            x1 = x1 - 1 if x1 - 1 > x0 else x1
    return (y0, y1, x0, x1), font


def assemble(episodes: list[Episode], fps: int, hold_s: float, label: str,
             crop, counter: bool = True) -> list[np.ndarray]:
    crop, font = fit_label(crop, episodes[0].frames[0].shape, label, len(episodes) if counter else 1)
    y0, y1, x0, x1 = crop
    hold = int(round(hold_s * fps))
    n = len(episodes)
    out = []
    for i, ep in enumerate(episodes):
        corner = f"episode {i + 1}/{n}" if counter and n > 1 else None
        for f in ep.frames:
            out.append(burn_label(f[y0:y1, x0:x1], label, corner, font))
        out.extend([out[-1]] * hold)
    return out, crop


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v3 as iio
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), np.stack(frames), plugin="pyav", codec="libx264",
                fps=fps, out_pixel_format="yuv420p")


def write_gif(path: Path, frames: list[np.ndarray], fps: int, max_frames: int = 120,
              max_width: int = 360) -> None:
    import imageio.v3 as iio
    from PIL import Image
    step = max(1, math.ceil(len(frames) / max_frames))
    sub = frames[::step]
    if sub[0].shape[1] > max_width:
        scale = max_width / sub[0].shape[1]
        size = (max_width, int(round(sub[0].shape[0] * scale)))
        sub = [np.asarray(Image.fromarray(f).resize(size, Image.LANCZOS)) for f in sub]
    # Pillow's GIF frame delay is in MILLISECONDS.
    iio.imwrite(str(path), np.stack(sub), plugin="pillow",
                duration=int(round(1000 * step / fps)), loop=0)


def write_still(path: Path, frame: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(frame).save(path)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return None


# ----------------------------------------------------------------------------
# one row
# ----------------------------------------------------------------------------
def render_row(row: VideoRow, out_mp4: Path, *, episodes: int, fps: int, hold: float,
               device: str, crop: str, still_frac: float, gif_max_frames: int,
               gif_width: int, dry_run: bool = False) -> dict:
    seed, why = pick_seed(row.tag, row.variant, row.seed)
    run = run_dir(row.tag, row.variant, seed)
    if not (run / "config.yaml").is_file():
        raise FileNotFoundError(f"{run}: no config.yaml")
    cfg = load_cfg(run)
    scenario = cfg.get("scenario_name")
    horizon = int(cfg.get("horizon", 100))
    sparse = bool(cfg.get("sparse_rewards", False))
    schema = ("binary" if (sparse and cfg.get("binary_terminal_reward", False))
              else "legacy_sparse" if sparse else "dense")
    entry = dict(
        task=row.task, tag=row.tag, variant=row.variant, label=row.label, slug=row.slug,
        seed=seed, seed_reason=why, run_dir=str(run.resolve()), protocol="rl",
        scenario=scenario, horizon=horizon, reward_schema=schema,
        env_seeds=[int(cfg.get("seed", 0)) + RENDER_SEED_OFFSET + i for i in range(episodes)],
    )
    if dry_run:
        return entry

    dev = torch.device(device)
    env = build_render_env(cfg, dev, seed)
    if row.variant.startswith("bc_eval"):
        policy, tpath, how = build_teacher_policy(cfg, env, dev)
        entry["teacher"] = dict(path=str(tpath), md5=md5(tpath), resolution=how)
    else:
        policy, ckpt = build_actor_policy(cfg, run, env, dev)
        entry["checkpoint"] = dict(path=str(ckpt.resolve()), md5=md5(ckpt))

    eps = [rollout_episode(env, policy, s, horizon, sparse) for s in entry["env_seeds"]]
    env.close()
    all_frames = [f for e in eps for f in e.frames]
    box = auto_crop_box(all_frames) if crop == "auto" else (0, all_frames[0].shape[0], 0, all_frames[0].shape[1])
    title = f"{TASK_TITLE.get(row.task, row.task)}: {row.label}"
    frames, box = assemble(eps, fps, hold, title, box)
    write_mp4(out_mp4, frames, fps)
    gif = out_mp4.with_suffix(".gif")
    write_gif(gif, frames, fps, gif_max_frames, gif_width)
    still = out_mp4.with_suffix(".png")
    y0, y1, x0, x1 = box
    k = min(len(eps[0].frames) - 1, int(still_frac * len(eps[0].frames)))
    write_still(still, eps[0].frames[k][y0:y1, x0:x1])

    entry.update(
        episodes=[dict(env_seed=e.env_seed, length=e.length, returns=e.returns,
                       goal_step=e.goal_step, terminated=e.terminated, truncated=e.truncated)
                  for e in eps],
        files=dict(mp4=str(out_mp4), gif=str(gif), still=str(still)),
        frame_size=[int(x1 - x0), int(y1 - y0)], crop=[int(v) for v in box],
        n_frames=len(frames), duration_s=round(len(frames) / fps, 2), fps=fps,
    )
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag")
    ap.add_argument("--variant")
    ap.add_argument("--seed", default="median", help="int, or 'median' (default)")
    ap.add_argument("--out", type=Path, help="output .mp4 (gif and png land beside it)")
    ap.add_argument("--label", default=None, help="text burned into the frames")
    ap.add_argument("--task", default=None, help="navigation | buzzwire | transport (title)")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold", type=float, default=0.7, help="seconds to hold each episode's last frame")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--crop", choices=["auto", "none"], default="auto")
    ap.add_argument("--still-frac", type=float, default=0.4)
    ap.add_argument("--gif-max-frames", type=int, default=120)
    ap.add_argument("--gif-width", type=int, default=360)
    ap.add_argument("--batch", choices=["poster"], default=None)
    ap.add_argument("--nav-tag", default="paper5")
    ap.add_argument("--bw-tag", default="buzzwire6")
    ap.add_argument("--tr-tag", default="transport800")
    ap.add_argument("--rows", type=Path, default=None,
                    help="JSON list of {task, tag, variant, label, slug[, seed]} replacing the poster table")
    ap.add_argument("--videos-dir", type=Path, default=ROOT / "poster" / "videos")
    ap.add_argument("--only", default=None, help="comma-separated slugs or variants to render (batch)")
    ap.add_argument("--overwrite", action="store_true", help="re-render rows whose mp4 exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    render_kw = dict(episodes=args.episodes, fps=args.fps, hold=args.hold, device=args.device,
                     crop=args.crop, still_frac=args.still_frac,
                     gif_max_frames=args.gif_max_frames, gif_width=args.gif_width,
                     dry_run=args.dry_run)

    if args.batch is None and args.rows is None:
        if not (args.tag and args.variant and args.out):
            ap.error("--tag, --variant and --out are required without --batch/--rows")
        task = args.task or {"navigation": "navigation", "buzz_wire": "buzzwire",
                             "transport": "transport"}.get(
            load_cfg(run_dir(args.tag, args.variant, pick_seed(args.tag, args.variant, args.seed)[0]))
            .get("scenario_name"), "")
        row = VideoRow(task, args.tag, args.variant, args.label or args.variant,
                       args.out.stem, args.seed)
        entry = render_row(row, args.out, **render_kw)
        print(json.dumps(entry, indent=2))
        return 0

    if args.rows:
        rows = [VideoRow(**r) for r in json.load(open(args.rows))]
    else:
        rows = poster_rows(args.nav_tag, args.bw_tag, args.tr_tag)
    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        rows = [r for r in rows if r.slug in keep or r.variant in keep]

    videos_dir = args.videos_dir
    videos_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = videos_dir / "manifest.json"
    manifest = dict(git_commit=git_commit(), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    fps=args.fps, episodes=args.episodes, rows=[])
    if manifest_path.exists():
        try:
            manifest["rows"] = json.load(open(manifest_path)).get("rows", [])
        except ValueError:
            pass
    by_key = {(r["task"], r["slug"]): r for r in manifest["rows"]}

    failures = 0
    for row in rows:
        out_mp4 = videos_dir / f"{row.task}_{row.slug}.mp4"
        key = (row.task, row.slug)
        t0 = time.time()
        try:
            if out_mp4.exists() and not args.overwrite and not args.dry_run:
                print(f"[skip] {out_mp4.name} exists (use --overwrite)")
                continue
            entry = render_row(row, out_mp4, **render_kw)
            entry["status"] = "dry-run" if args.dry_run else "ok"
            entry["render_seconds"] = round(time.time() - t0, 1)
            print(f"[{'plan' if args.dry_run else 'ok'}] {row.task}/{row.slug}: {row.tag}/{row.variant} "
                  f"seed {entry['seed']} ({entry['seed_reason'].split(' [')[0]})"
                  + ("" if args.dry_run else f" -> {out_mp4.name} {entry['n_frames']} frames "
                     f"{entry['duration_s']} s in {entry['render_seconds']} s"))
        except FileNotFoundError as e:
            entry = dict(task=row.task, tag=row.tag, variant=row.variant, label=row.label,
                         slug=row.slug, status="missing", error=str(e))
            print(f"[missing] {row.task}/{row.slug}: {e}")
        except Exception as e:  # keep going; the manifest records the failure
            failures += 1
            entry = dict(task=row.task, tag=row.tag, variant=row.variant, label=row.label,
                         slug=row.slug, status="failed", error=repr(e))
            print(f"[FAILED] {row.task}/{row.slug}: {e!r}")
            import traceback
            traceback.print_exc()
        by_key[key] = entry
        if not args.dry_run:
            manifest["rows"] = [by_key[k] for k in sorted(by_key)]
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
    if not args.dry_run:
        print(f"manifest: {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
