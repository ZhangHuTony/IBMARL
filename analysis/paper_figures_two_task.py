"""
Two-task figures for the workshop paper (Navigation = results/paper3,
Buzzwire = results/buzzwire3), written to paper/ next to the LaTeX source.

    python -m analysis.paper_figures_two_task            # both figures
    python -m analysis.paper_figures_two_task --frame 90 # pick the Buzzwire frame

fig_results.{pdf,png}  2 x 2 learning curves: columns = tasks, top row =
                       IBMARL vs baselines, bottom row = arbiter variants.
                       Navigation curves use the paper3 preset (9-eval window);
                       Buzzwire uses the buzzwire3 preset (3-eval window) and
                       the executed-policy protocol (IBMARL's arbitrated RL+IL
                       policy, the baselines' own actors).
fig_tasks.{pdf,png}    the two environments: paper/fig_env_navigation.png
                       (adapted from R2BC) and a frame of the R2BC Buzzwire
                       demonstration GIF cropped to the corridor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis import paper3_figures as pf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper"
NAV_PNG = OUT / "fig_env_navigation.png"
BW_GIF = Path("/home/tony-zhang/Research/ibmarl/R2BC/results/"
              "buzzwire_r2bc_decent_20260826_225653/performance_demo.gif")

TASKS = [  # (title, results tag, smoothing window, protocol)
    ("Navigation", "paper3", 9, "rl"),
    ("Buzzwire", "buzzwire3", 3, "executed"),
]
ROWS = [  # (row label, variants, short legend labels)
    ("baselines", pf.BASELINES,
     {"ibmarl_strict": "IBMARL", "rlfd": "RLfD", "rft": "RFT", "maddpg": "MADDPG"}),
    ("arbiter variants", pf.ABLATIONS,
     {"ibmarl_strict": "IBMARL", "ibmarl": "w/ per-agent mixing",
      "ibmarl_strict_hard": "w/o soft selection",
      "ibmarl_strict_1critic": "w/o critic ensemble"}),
]


def fig_results(height: float = 2.6) -> None:
    """2 x 2 learning curves with one horizontal legend under each row (the
    layout of IBRL's Fig. 4).  Panels in a column share the x-range, so the
    top row's tick labels are dropped."""
    fig, axes = plt.subplots(2, 2, figsize=(pf.COL_W, height))
    for col, (title, tag, smooth, protocol) in enumerate(TASKS):
        pf.configure(tag, smooth, protocol)
        il = pf.bc_level()
        for row, (_, variants, short) in enumerate(ROWS):
            ax = axes[row, col]
            for v in variants:
                x, y = pf.curve(v, "eval_reward_mean")
                m, sem = pf.mean_sem(y)
                ax.plot(x, m, color=pf.C[v], label=short[v], zorder=3, lw=1.1)
                ax.fill_between(x, m - sem, m + sem, color=pf.C[v], alpha=0.18,
                                lw=0, zorder=2)
            ax.axhline(il, color=pf.C["bc"], ls=(0, (4, 2.5)), lw=0.9, zorder=1,
                       label="IL policy")
            pf.style_return_axis(ax, title=title if row == 0 else None,
                                 ylabel=(col == 0), xlabel=(row == 1))
            if row == 0:
                ax.tick_params(labelbottom=False)
            ax.text(0.975, 0.96, "(%s)" % "abcd"[row * 2 + col],
                    transform=ax.transAxes, va="top", ha="right", fontsize=7.5)
    left, right = 0.135, 0.965
    fig.subplots_adjust(left=left, right=right, top=0.93, bottom=0.18,
                        wspace=0.24, hspace=0.28)
    # Legends: one row per panel row.  The top one sits in the gap between
    # the rows (the top row has no tick labels); the bottom one goes under
    # the measured bounding box of the x-axis label, using the full figure
    # width because the variant names are long.
    kw = dict(frameon=False, handlelength=1.4, handletextpad=0.4,
              borderpad=0.0, borderaxespad=0.0)
    top, bot = axes[0, 0].get_position(), axes[1, 0].get_position()
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="center", fontsize=6.4, columnspacing=1.0,
               bbox_to_anchor=((left + right) / 2, (top.y0 + bot.y1) / 2),
               ncol=len(h), **kw)
    fig.canvas.draw()
    bb = axes[1, 0].xaxis.label.get_window_extent(fig.canvas.get_renderer())
    y_label_bottom = fig.transFigure.inverted().transform((0, bb.y0))[1]
    h, l = axes[1, 0].get_legend_handles_labels()
    h, l = h[:-1], l[:-1]               # the IL line is named in the top legend
    fig.legend(h, l, loc="upper center", fontsize=6.0, columnspacing=0.8,
               bbox_to_anchor=(0.5, y_label_bottom - 0.012), ncol=len(h), **kw)
    fig.savefig(OUT / "fig_results.pdf", facecolor="white")
    fig.savefig(OUT / "fig_results.png", dpi=pf.PNG_DPI, facecolor="white")
    print(f"wrote {OUT/'fig_results.pdf'}  ({pf.COL_W:.2f} x {height:.2f} in)")


def buzzwire_frame(index: int, pad: float = 0.35) -> np.ndarray:
    """Fallback: one R2BC demo-GIF frame cropped to the corridor."""
    gif = Image.open(BW_GIF)
    n = gif.n_frames
    ys, xs = [], []
    for k in range(0, n, 10):
        gif.seek(k)
        a = np.asarray(gif.convert("L"))
        yy, xx = np.nonzero(a < 245)
        ys += [yy.min(), yy.max()]; xs += [xx.min(), xx.max()]
    y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
    w = x1 - x0
    x0, x1 = int(x0 - pad * w), int(x1 + pad * w)
    y0, y1 = int(y0 - 0.06 * (y1 - y0)), int(y1 + 0.06 * (y1 - y0))
    gif.seek(index)
    return np.asarray(gif.convert("RGB"))[y0:y1, x0:x1]


def buzzwire_render(seed: int = 3, zoom: float = 0.75, t_stop: int = 70) -> np.ndarray:
    """Render the sparse Buzzwire scenario itself (needs a display), driving
    the team with the demonstrator's rule (hold a lateral offset, push toward
    the goal's y) for `t_stop` steps, and crop to the goal..ball segment so
    the agents are legible at column width.  The corridor walls run through
    the crop; its ends are not shown."""
    import torch
    from vmas import make_env
    from src.environment.scenarios.buzz_wire_sparse import SparseRewardBuzzWireScenario
    sc = SparseRewardBuzzWireScenario()
    sc.viewer_zoom = zoom
    sc.viewer_size = (420, 760)
    env = make_env(scenario=sc, num_envs=1, device="cpu",
                   continuous_actions=True, seed=seed)
    obs = env.reset()
    for _ in range(t_stop):
        acts = []
        for o in obs:
            x = o[0, 0].item()
            sign = 1.0 if x >= 0 else -1.0
            ax = 4.0 * (sign * 0.5 - x)
            ay = 4.0 * (-(o[0, 5].item()))
            acts.append(torch.tensor([[max(-1, min(1, ax)), max(-1, min(1, ay))]]))
        obs, *_ = env.step(acts)
    img = np.asarray(env.render(mode="rgb_array", visualize_when_rgb=False))
    a = img.astype(int)
    sat = a.max(-1) - a.min(-1)          # coloured pixels: goal (green), agents (purple)
    yy, xx = np.nonzero(sat > 60)
    y0, y1, x0, x1 = yy.min(), yy.max(), xx.min(), xx.max()
    py, px = int(0.18 * (y1 - y0)), int(0.12 * (x1 - x0))
    return img[max(0, y0 - py):y1 + py, max(0, x0 - px):x1 + px]


def fig_tasks(frame: int, height: float = 1.1) -> None:
    nav = np.asarray(Image.open(NAV_PNG).convert("RGB"))
    try:
        bw = buzzwire_render()
    except Exception as e:  # headless box: fall back to the demo GIF
        print(f"  .. render failed ({e!r}); using GIF frame {frame}")
        bw = buzzwire_frame(frame)
    nav_w = height * nav.shape[1] / nav.shape[0]
    bw_w = height * bw.shape[1] / bw.shape[0]
    fig_h = height + 0.3            # room above the images for the panel titles
    fig = plt.figure(figsize=(pf.COL_W, fig_h))
    gap = 0.35
    total = nav_w + bw_w + gap
    x = (pf.COL_W - total) / 2
    for img, w, label in ((nav, nav_w, "(a) Navigation"), (bw, bw_w, "(b) Buzzwire")):
        ax = fig.add_axes([x / pf.COL_W, 0.02, w / pf.COL_W, height / fig_h])
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(label, fontsize=8, pad=2)
        x += w + gap
    fig.savefig(OUT / "fig_tasks.pdf", facecolor="white")
    fig.savefig(OUT / "fig_tasks.png", dpi=pf.PNG_DPI, facecolor="white")
    print(f"wrote {OUT/'fig_tasks.pdf'}  (frame {frame}, crop {bw.shape[1]}x{bw.shape[0]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", type=int, default=90, help="Buzzwire GIF frame")
    ap.add_argument("--only", choices=["results", "tasks"], default=None)
    args = ap.parse_args()
    if args.only in (None, "results"):
        fig_results()
    if args.only in (None, "tasks"):
        fig_tasks(args.frame)
