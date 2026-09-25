"""
Poster figures: one panel per task, each assembled from whichever tags hold
that task's arms, in a poster-sized style.

    python -m analysis.poster_figures                       # everything into poster/
    python -m analysis.poster_figures --skip-missing        # while arms are still running
    python -m analysis.poster_figures --only table,readme
    python -m analysis.poster_figures --spec my_rows.json   # custom (tag, variant) rows

Outputs (poster/):
  fig_main.{pdf,png}      Navigation | Buzz-wire | Transport: MADDPG, RLfD, RFT,
                          IBMARL and the R2BC teacher line, every method
                          evaluated with its RL actors alone.
  fig_ablation.{pdf,png}  IBMARL (both protocols), w/o gated term (both), w/o
                          mixing (RL actors), teacher line.
  fig_human.{pdf,png}     Buzz-wire with the human Xbox teacher: IBMARL (both
                          protocols), RFT, RLfD, MADDPG, human-teacher line.
  fig_tasks.{pdf,png}     Three environment stills from poster/videos/*.png.
  summary_table.{md,csv}  Final-10 return per (task, method, protocol), steps to
                          teacher, teacher levels, seed counts, source tags.
  README.md               Sources and caveats, so the folder explains itself.

Curves, smoothing, colours and the protocol column mapping come from
analysis/paper3_figures.py (imported, not copied); the Table-I statistics from
analysis/steps_to_teacher.py.  Nothing here touches that module's globals:
every read names its tag explicitly, which is what lets one panel mix tags.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis import paper3_figures as pf
from analysis.steps_to_teacher import variant_stats

ROOT = pf.ROOT


@dataclass(frozen=True)
class Row:
    tag: str
    variant: str
    label: str
    protocols: tuple = ("rl",)     # "rl" = RL actors only, "executed" = arbitrated RL+IL


@dataclass
class TaskSpec:
    key: str                       # navigation | buzzwire | transport (file-name key)
    title: str
    preset: str                    # pf.PRESETS key: axes and smoothing window
    teacher: tuple                 # (tag, bc_eval variant)
    main: list = field(default_factory=list)
    ablation: list = field(default_factory=list)
    human: list = field(default_factory=list)
    human_teacher: tuple | None = None


def default_tasks(nav_tag: str, bw_tag: str, tr_tag: str) -> list[TaskSpec]:
    """
    The poster's rows.  The poster tags hold every arm of a task, the reused
    ones as symlinks into the sweep that produced them (paper3/paper4 for
    navigation, buzzwire1/3/3_reg for buzz-wire), so one tag per task suffices;
    a --spec file can point any row elsewhere.
    """
    both = ("rl", "executed")
    tasks = []
    for key, title, preset, tag in (("navigation", "Navigation", "paper3", nav_tag),
                                    ("buzzwire", "Buzz-wire", "buzzwire3", bw_tag),
                                    ("transport", "Transport", "transport800", tr_tag)):
        t = TaskSpec(key, title, preset, (tag, "bc_eval"))
        # Main comparison: every method under the same protocol, its RL actors
        # alone -- the executed RL+IL curve lives in the ablation figure.
        t.main = [Row(tag, "maddpg", "MADDPG"), Row(tag, "rlfd", "RLfD"), Row(tag, "rft", "RFT"),
                  Row(tag, "ibmarl_gated_a0.4", "IBMARL")]
        t.ablation = [Row(tag, "ibmarl_gated_a0.4", "IBMARL", both),
                      Row(tag, "ibmarl", "w/o gated term", both),
                      Row(tag, "ibmarl_strict_gated_a0.4", "w/o mixing")]
        if key == "buzzwire":
            t.human = [Row(tag, "maddpg", "MADDPG"), Row(tag, "rlfd_human", "RLfD"),
                       Row(tag, "rft_human", "RFT"),
                       Row(tag, "ibmarl_gated_a0.4_human", "IBMARL", both)]
            t.human_teacher = (tag, "bc_eval_human")
        tasks.append(t)
    return tasks


def tasks_from_spec(path: Path) -> list[TaskSpec]:
    spec = json.load(open(path))
    tasks = []
    for t in spec:
        rows = {k: [Row(r["tag"], r["variant"], r["label"], tuple(r.get("protocols", ["rl"])))
                    for r in t.get(k, [])] for k in ("main", "ablation", "human")}
        tasks.append(TaskSpec(t["key"], t["title"], t["preset"], tuple(t["teacher"]),
                              rows["main"], rows["ablation"], rows["human"],
                              tuple(t["human_teacher"]) if t.get("human_teacher") else None))
    return tasks


# --- style ------------------------------------------------------------------
POSTER_C = {
    "MADDPG": pf.C["maddpg"],
    "RLfD": pf.C["rlfd"],
    "RFT": pf.C["rft"],
    "IBMARL": pf.C["ibmarl_gated_a0.4"],
    "w/o gated term": pf.C["ibmarl"],
    "w/o mixing": pf.C["ibmarl_strict_gated_a0.4"],
    "teacher": pf.C["bc"],
}
LS = {"rl": "-", "executed": (0, (4, 2))}
BAND_ALPHA = {"rl": 0.18, "executed": 0.10}
# paper3_figures applies 8 pt IEEE rcParams at import; the poster overrides
# them inside an rc_context so nothing leaks into that module's own figures.
POSTER_RC = {
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 15,
    "lines.linewidth": 3.0, "axes.linewidth": 1.2,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "legend.frameon": False, "savefig.bbox": "tight",
}


def color_of(label: str) -> str:
    return POSTER_C.get(label, "0.3")


# --- data -------------------------------------------------------------------
def preset_of(task: TaskSpec) -> dict:
    return pf.PRESETS.get(task.preset, pf._EMPTY_PRESET)


def row_curve(row: Row, protocol: str, smooth: int):
    return pf.curve(row.variant, "eval_reward_mean", smooth=smooth, protocol=protocol,
                    run=row.tag, complete_only=True)


def teacher_level(tag: str, variant: str) -> tuple[float, float, int]:
    vals = np.array(pf.bc_values(run=tag, variant=variant))
    sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return float(vals.mean()), sem, len(vals)


def n_seeds(row: Row) -> int:
    return len(pf.seed_dirs(row.variant, row.tag, complete_only=True))


# --- panels -----------------------------------------------------------------
def style_axis(ax, task: TaskSpec, ylabel: bool = True):
    p = preset_of(task)
    ax.set_title(task.title, pad=8)
    if p.get("xlim"):
        ax.set_xlim(*p["xlim"])
    if p.get("ylim"):
        ax.set_ylim(*p["ylim"])
    if p.get("xticks"):
        ax.set_xticks(p["xticks"])
    if p.get("yticks"):
        ax.set_yticks(p["yticks"])
    ax.set_xlabel(r"Interaction steps ($\times$1000)")
    if ylabel:
        ax.set_ylabel(p.get("ylabel") or pf.RETURN_LABEL)
    ax.spines[["top", "right"]].set_visible(False)


def draw_panel(ax, task: TaskSpec, rows: list[Row], teacher: tuple, teacher_label: str,
               skip_missing: bool):
    """Curves for *rows* plus the teacher line; returns (legend handles, protocols drawn, missing labels)."""
    smooth = preset_of(task).get("smooth", 9)
    handles, protocols, missing = {}, [], []
    for row in rows:
        for protocol in row.protocols:
            try:
                x, y = row_curve(row, protocol, smooth)
            except FileNotFoundError:
                if not skip_missing:
                    raise
                if row.label not in missing:
                    missing.append(row.label)
                    print(f"  .. {task.key}: no runs for {row.tag}/{row.variant} ({row.label}); skipped")
                continue
            m, sem = pf.mean_sem(y)
            c = color_of(row.label)
            ax.plot(x, m, color=c, ls=LS[protocol], zorder=3)
            ax.fill_between(x, m - sem, m + sem, color=c, alpha=BAND_ALPHA[protocol],
                            lw=0, zorder=2)
            handles.setdefault(row.label, Line2D([], [], color=c, lw=3.0, label=row.label))
            if protocol not in protocols:
                protocols.append(protocol)
    try:
        level, _, _ = teacher_level(*teacher)
        ax.axhline(level, color=POSTER_C["teacher"], ls=(0, (4, 2.5)), lw=2.2, zorder=1)
        handles[teacher_label] = Line2D([], [], color=POSTER_C["teacher"], ls=(0, (4, 2.5)),
                                        lw=2.2, label=teacher_label)
    except FileNotFoundError:
        if not skip_missing:
            raise
        missing.append(teacher_label)
    if missing:
        ax.text(0.02, 0.03, "pending: " + ", ".join(missing), transform=ax.transAxes,
                fontsize=11, color="#B03030", va="bottom", ha="left", zorder=5)
    return handles, protocols, missing


LEGEND_ORDER = ["IBMARL", "w/o gated term", "w/o mixing", "MADDPG", "RLfD", "RFT",
                "R2BC (teacher)", "Human teacher (R2BC)"]


def shared_legend(fig, handles: dict, protocols: list, bottom: float):
    # The headline method first, whatever order the panels produced the
    # handles in (a panel whose IBMARL arm is still pending yields it last).
    rank = {k: i for i, k in enumerate(LEGEND_ORDER)}
    items = [handles[k] for k in sorted(handles, key=lambda k: (rank.get(k, 99), k))]
    if len(protocols) > 1:
        for pr in ("rl", "executed"):
            if pr in protocols:
                label, ls = pf.PROTOCOL_LEGEND[pr]
                items.append(Line2D([], [], color="0.3", ls=ls, lw=2.6, label=label))
    fig.legend(handles=items, loc="lower center", ncol=min(len(items), 7),
               bbox_to_anchor=(0.5, bottom), handlelength=2.4, columnspacing=1.6)


def save(fig, out: Path, name: str, png_dpi: int):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", facecolor="white")
    fig.savefig(out / f"{name}.png", dpi=png_dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote {out / name}.pdf / .png")


def fig_panels(tasks: list[TaskSpec], which: str, out: Path, name: str, skip_missing: bool,
               png_dpi: int):
    with plt.rc_context(POSTER_RC):
        fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 5.4))
        axes = np.atleast_1d(axes)
        all_handles, all_protocols = {}, []
        for ax, task in zip(axes, tasks):
            rows = getattr(task, which)
            handles, protocols, _ = draw_panel(ax, task, rows, task.teacher, "R2BC (teacher)",
                                               skip_missing)
            style_axis(ax, task, ylabel=True)
            for k, v in handles.items():
                all_handles.setdefault(k, v)
            for pr in protocols:
                if pr not in all_protocols:
                    all_protocols.append(pr)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        shared_legend(fig, all_handles, all_protocols, bottom=0.0)
        save(fig, out, name, png_dpi)


def fig_human(tasks: list[TaskSpec], out: Path, skip_missing: bool, png_dpi: int):
    human = [t for t in tasks if t.human]
    if not human:
        print("  .. fig_human: no task has a human-teacher arm; skipped")
        return
    with plt.rc_context(POSTER_RC):
        fig, axes = plt.subplots(1, len(human), figsize=(5.6 * len(human), 5.4))
        axes = np.atleast_1d(axes)
        all_handles, all_protocols = {}, []
        for ax, task in zip(axes, human):
            handles, protocols, _ = draw_panel(ax, task, task.human, task.human_teacher,
                                               "Human teacher (R2BC)", skip_missing)
            style_axis(ax, task)
            ax.set_title(f"{task.title}, human demonstrations", pad=8)
            all_handles.update({k: v for k, v in handles.items() if k not in all_handles})
            all_protocols += [p for p in protocols if p not in all_protocols]
        fig.tight_layout(rect=(0, 0.16, 1, 1))
        shared_legend(fig, all_handles, all_protocols, bottom=0.0)
        save(fig, out, "fig_human", png_dpi)


def load_still(videos_dir: Path, task_key: str, prefer=("ibmarl", "teacher")):
    from PIL import Image
    for slug in prefer:
        p = videos_dir / f"{task_key}_{slug}.png"
        if p.is_file():
            return np.asarray(Image.open(p).convert("RGB")), p
    for p in sorted(videos_dir.glob(f"{task_key}_*.png")):
        return np.asarray(Image.open(p).convert("RGB")), p
    return None, None


def fig_tasks(tasks: list[TaskSpec], videos_dir: Path, out: Path, png_dpi: int, height: float = 4.2):
    stills = []
    for t in tasks:
        img, p = load_still(videos_dir, t.key)
        if img is None:
            print(f"  .. fig_tasks: no still for {t.key} under {videos_dir} (render the videos first); skipped")
            return
        stills.append((t, img))
    widths = [height * img.shape[1] / img.shape[0] for _, img in stills]
    gap = 0.4
    fig_w = sum(widths) + gap * (len(stills) + 1)
    fig_h = height + 0.7
    with plt.rc_context(POSTER_RC):
        fig = plt.figure(figsize=(fig_w, fig_h))
        x = gap
        for (t, img), w in zip(stills, widths):
            ax = fig.add_axes([x / fig_w, 0.02, w / fig_w, height / fig_h])
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(t.title, pad=6)
            x += w + gap
        save(fig, out, "fig_tasks", png_dpi)


# --- table and README -------------------------------------------------------
def summary_rows(tasks: list[TaskSpec], skip_missing: bool) -> list[dict]:
    rows = []
    for task in tasks:
        groups = [("main", task.main, task.teacher, "R2BC (teacher)")]
        if task.human:
            groups.append(("human", task.human, task.human_teacher, "Human teacher (R2BC)"))
        seen = set()
        for group, grp_rows, teacher, teacher_label in groups:
            try:
                level, sem, n = teacher_level(*teacher)
            except FileNotFoundError:
                if not skip_missing:
                    raise
                level = float("nan"); sem = float("nan"); n = 0
            rows.append(dict(task=task.title, group=group, method=teacher_label, protocol="teacher",
                             tag=teacher[0], variant=teacher[1], n_seeds=n,
                             final_mean=level, final_sem=sem, steps_to_teacher_k=None,
                             n_reached=None))
            for row in list(grp_rows) + (list(task.ablation) if group == "main" else []):
                for protocol in row.protocols:
                    key = (row.tag, row.variant, protocol)
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        st = variant_stats(row.tag, row.variant, level, smooth=preset_of(task).get("smooth"),
                                           protocol=protocol, complete_only=True)
                    except FileNotFoundError:
                        if not skip_missing:
                            raise
                        rows.append(dict(task=task.title, group=group, method=row.label, protocol=protocol,
                                         tag=row.tag, variant=row.variant, n_seeds=0, final_mean=None,
                                         final_sem=None, steps_to_teacher_k=None, n_reached=None,
                                         status="pending"))
                        continue
                    rows.append(dict(
                        task=task.title, group=group, method=row.label, protocol=protocol,
                        tag=row.tag, variant=row.variant, n_seeds=st["n_seeds"],
                        final_mean=st["final_return_mean"], final_sem=st["final_return_sem"],
                        steps_to_teacher_k=st["steps_to_teacher_median"] / 1000.0,
                        n_reached=st["n_reached"], capped=st["capped_at_budget"],
                        finals=st["finals"]))
    return rows


def write_table(rows: list[dict], out: Path):
    import pandas as pd
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.drop(columns=[c for c in ("finals",) if c in df.columns]).to_csv(out / "summary_table.csv", index=False)
    lines = ["| task | method | protocol | n | final (last 10 evals) | steps to teacher | source |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("status") == "pending":
            proto = pf.PROTOCOL_LEGEND[r["protocol"]][0]
            lines.append(f"| {r['task']} | {r['method']} | {proto} | 0 | pending | – | {r['tag']}/{r['variant']} |")
            continue
        if r["protocol"] == "teacher":
            final = f"{r['final_mean']:.2f} ± {r['final_sem']:.2f}" if r["n_seeds"] else "pending"
            lines.append(f"| {r['task']} | {r['method']} | – | {r['n_seeds']} | {final} | – | {r['tag']}/{r['variant']} |")
            continue
        proto = pf.PROTOCOL_LEGEND[r["protocol"]][0]
        stt = (f"{r['steps_to_teacher_k']:.0f}k" + ("†" if r.get("capped") else "")
               + f" ({r['n_reached']}/{r['n_seeds']})")
        lines.append(f"| {r['task']} | {r['method']} | {proto} | {r['n_seeds']} | "
                     f"{r['final_mean']:.2f} ± {r['final_sem']:.2f} | {stt} | {r['tag']}/{r['variant']} |")
    lines.append("")
    lines.append("final = mean ± s.e.m. over seeds of each seed's last-10-evaluation mean; "
                 "steps to teacher = median over seeds of the first evaluation whose smoothed curve "
                 "exceeds the teacher level († = at least one seed never did and is capped at the budget).")
    (out / "summary_table.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'summary_table.md'} / .csv ({len(rows)} rows)")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_readme(tasks: list[TaskSpec], out: Path, videos_dir: Path):
    manifest = videos_dir / "manifest.json"
    lines = [
        "# Poster figures and videos",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python -m analysis.poster_figures` at commit `{git_commit()}`.",
        "Videos by `python -m analysis.render_policy_videos --batch poster` (see `videos/manifest.json`).",
        "",
        "## Sources",
        "",
        "Each panel is assembled from the poster tag of its task; the reused arms are symlinks into",
        "the sweep that produced them (results/<tag>/<variant> -> ../<sweep>/<variant>).",
        "",
    ]
    for t in tasks:
        lines.append(f"### {t.title} (`{t.main[0].tag}`, axes/smoothing preset `{t.preset}`)")
        lines.append("")
        lines.append("| arm | tag/variant | seeds | resolves to |")
        lines.append("|---|---|---|---|")
        seen = set()
        for row in t.main + t.ablation + t.human:
            if (row.tag, row.variant) in seen:
                continue
            seen.add((row.tag, row.variant))
            d = ROOT / "results" / row.tag / row.variant
            target = d.resolve().relative_to(ROOT) if d.exists() else "(not on disk yet)"
            n = n_seeds(row) if d.exists() else 0
            lines.append(f"| {row.label} | `{row.tag}/{row.variant}` | {n} | `{target}` |")
        for teacher, name in ((t.teacher, "R2BC teacher"), (t.human_teacher, "human teacher")):
            if teacher is None:
                continue
            try:
                level, sem, n = teacher_level(*teacher)
                lines.append(f"| {name} line | `{teacher[0]}/{teacher[1]}` | {n} | level {level:.2f} ± {sem:.2f} |")
            except FileNotFoundError:
                lines.append(f"| {name} line | `{teacher[0]}/{teacher[1]}` | 0 | pending |")
        lines.append("")
    lines += [
        "## Protocols",
        "",
        "Solid curves are the RL actors alone (deterministic, no teacher or arbiter in the loop),",
        "the protocol every baseline is measured under.  Dashed curves are IBMARL's executed",
        "policy: the arbiter choosing greedily between the teacher's and the RL actors' proposals",
        "(IBRL's reporting convention).  Bands are ± s.e.m. over seeds.",
        "",
        "## Caveats",
        "",
        "- Navigation and Buzz-wire use the legacy -1/step sparse reward (return = -(steps off goal);",
        "  floor -100 / -200); Transport uses VMAS's dense reward.",
        "- The reused baselines (paper3 for Navigation; buzzwire3, with MADDPG from buzzwire1, for",
        "  Buzz-wire) were trained before the 2026-09-17 eval-guard fix (noise only) and before the",
        "  2026-09-21 baseline changes (target actor, full-run RFT anneal); the human-teacher RFT/RLfD",
        "  arms use the current code.  IBMARL's no-gate arm on Navigation is paper4 (Aug 27 code) plus",
        "  one seed rerun from the same frozen config.",
        "- The IBMARL arms run from frozen copies of those sweeps' configs (config/legacy/), so",
        "  teacher, demos, replay size, warm-up, arbiter temperature (0.05) and eval cadence match",
        "  the runs they are drawn beside.",
        "- Videos show one representative seed per arm (the median final-10 seed) for 3 deterministic",
        "  episodes; see videos/manifest.json for the seed, checkpoint md5 and per-episode returns.",
        "",
    ]
    if manifest.is_file():
        try:
            rows = json.load(open(manifest)).get("rows", [])
            lines.append("## Videos")
            lines.append("")
            lines.append("| file | tag/variant | seed | episodes (return) |")
            lines.append("|---|---|---|---|")
            for r in rows:
                if r.get("status") != "ok":
                    lines.append(f"| {r['task']}_{r['slug']} | `{r['tag']}/{r['variant']}` | – | {r.get('status')} |")
                    continue
                eps = ", ".join(f"{e['length']} steps ({list(e['returns'].values())[0]:.1f})" for e in r["episodes"])
                lines.append(f"| {Path(r['files']['mp4']).name} | `{r['tag']}/{r['variant']}` | {r['seed']} | {eps} |")
            lines.append("")
        except ValueError:
            pass
    (out / "README.md").write_text("\n".join(lines))
    print(f"wrote {out / 'README.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nav-tag", default="paper5")
    ap.add_argument("--bw-tag", default="buzzwire6")
    ap.add_argument("--tr-tag", default="transport800")
    ap.add_argument("--out", type=Path, default=ROOT / "poster")
    ap.add_argument("--videos-dir", type=Path, default=None, help="default <out>/videos")
    ap.add_argument("--spec", type=Path, default=None, help="JSON task spec replacing the defaults")
    ap.add_argument("--only", default="main,ablation,human,tasks,table,readme")
    ap.add_argument("--skip-missing", action="store_true",
                    help="annotate arms that have no runs yet instead of failing")
    ap.add_argument("--png-dpi", type=int, default=300)
    args = ap.parse_args()

    tasks = tasks_from_spec(args.spec) if args.spec else default_tasks(args.nav_tag, args.bw_tag, args.tr_tag)
    videos_dir = args.videos_dir or (args.out / "videos")
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    args.out.mkdir(parents=True, exist_ok=True)

    if "main" in only:
        fig_panels(tasks, "main", args.out, "fig_main", args.skip_missing, args.png_dpi)
    if "ablation" in only:
        fig_panels(tasks, "ablation", args.out, "fig_ablation", args.skip_missing, args.png_dpi)
    if "human" in only:
        fig_human(tasks, args.out, args.skip_missing, args.png_dpi)
    if "tasks" in only:
        fig_tasks(tasks, videos_dir, args.out, args.png_dpi)
    if "table" in only:
        write_table(summary_rows(tasks, args.skip_missing), args.out)
    if "readme" in only:
        write_readme(tasks, args.out, videos_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
