"""
Refit a behaviour-cloning teacher on a SUBSET of a bundled recording.

Contingency for the balance audit (analysis/balance_teacher_audit.py): if the
bundled 36-demo teacher reaches the basin in more than ~60% of episodes at
every sensible radius, the sweep has no headroom above the teacher line and the
teacher has to be weakened.  The R2BC trainer that produced the bundled
checkpoints is not in this repository (it needs the MAPPO supervisor on the
collaborator's machine; see origin/feat-in-suite-r2bc), so this fits the same
network -- DecentralizedMiniBC from src/r2bc/mabc.py, the class
R2bcPolicy loads -- on the first k episodes of the recording with the same
objective R2BC uses (MSE on the clamped action, Adam 1e-3, batch 256, stop
when the epoch loss stops improving).

Caveat, recorded in metadata.json: this is BC on a subset of R2BC's recording,
not a smaller R2BC run.  R2BC's demos are a round-robin mixture (one agent
driven by the supervisor per episode while the others run the BC policies of
that round), so the first k episodes are also the EARLIEST rounds, where the BC
teammates were weakest -- which is the direction wanted here.

Writes teachers/<task>_<k>demo/{policy_checkpoint.pth, demonstrations.pt,
metadata.json}.  demonstrations.pt is the subset in the raw recording format
(dense rewards, recorder's on-goal flag as ``dones``), episode-contiguous rows,
so analysis/relabel_demos_sparse.py relabels it like the original (it infers an
env interleave of 1).  Then re-run the audit on the new checkpoint and, if it is
in band, point the bc_eval/ibmarl/rlfd/rft balance overlays at the new folder.

    python -m analysis.train_bc_subset balance --n-demos 18 --seed 0
    python -m analysis.train_bc_subset balance --episode-ids 0,3,5,...  # explicit choice
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from analysis.relabel_demos_binary import infer_env_count, to_episodes
from src.r2bc.mabc import DecentralizedMiniBC
from src.util.paths import PROJECT_ROOT

HORIZON = {"balance": 300, "navigation": 100, "buzz_wire": 200, "transport": 500}


def load_episodes(src: Path, horizon: int) -> tuple[dict[str, np.ndarray], int]:
    d = torch.load(src, map_location="cpu", weights_only=False)
    rec = {
        "obs": np.asarray(d["obs"], dtype=np.float32),
        "act": np.asarray(d["act"], dtype=np.float32),
        "rewards": np.asarray(d["rewards"], dtype=np.float32),
        "next_obs": np.asarray(d["next_obs"], dtype=np.float32),
        "dones": np.asarray(d["dones"], dtype=bool),
    }
    if rec["rewards"].ndim == 3:
        rec["rewards"] = rec["rewards"][..., 0]
    n_envs, cont = infer_env_count(rec["obs"], rec["next_obs"], horizon)
    ep = {k: to_episodes(v, n_envs, horizon) for k, v in rec.items()}
    print(f"{src}: {rec['obs'].shape[0]} rows -> {ep['obs'].shape[0]} episodes of {horizon} "
          f"(env interleave {n_envs}, continuity {cont:.4f})")
    return ep, n_envs


def fit(obs: np.ndarray, act: np.ndarray, *, hidden: int, layers: int, lr: float, batch: int,
        max_epochs: int, patience: int, min_delta: float, seed: int, device: torch.device) -> tuple[DecentralizedMiniBC, dict]:
    n, a, o = obs.shape
    act_dim = act.shape[-1]
    torch.manual_seed(seed)
    model = DecentralizedMiniBC(n=a, in_size=a * o, out_size=a * act_dim,
                                hidden_size=hidden, hidden_layers=layers).to(device)
    x = torch.from_numpy(obs.reshape(n, a * o)).to(device)
    y = torch.from_numpy(act.reshape(n, a * act_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    g = torch.Generator(device="cpu").manual_seed(seed)

    best, since_best, history = float("inf"), 0, []
    for epoch in range(max_epochs):
        perm = torch.randperm(n, generator=g).to(device)
        losses = []
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(torch.clamp(model(x[idx]), -1, 1), y[idx])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        mean = float(np.mean(losses))
        history.append(mean)
        if best - mean > min_delta:
            best, since_best = mean, 0
        else:
            since_best += 1
        if epoch % 10 == 0 or since_best == 0 and epoch < 10:
            print(f"  epoch {epoch:4d}  loss {mean:.5f}  (best {best:.5f})")
        if since_best >= patience:
            print(f"  stop at epoch {epoch}: no improvement > {min_delta} for {patience} epochs")
            break
    model.eval()
    with torch.no_grad():
        final = loss_fn(torch.clamp(model(x), -1, 1), y).item()
    return model, dict(epochs=len(history), final_loss=round(final, 6), best_epoch_loss=round(best, 6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=sorted(HORIZON))
    ap.add_argument("--src", type=Path, default=None, help="default teachers/<task>/demonstrations.pt")
    ap.add_argument("--n-demos", type=int, default=None, help="take the first k episodes")
    ap.add_argument("--episode-ids", default=None, help="comma-separated episode indices instead of --n-demos")
    ap.add_argument("--out", type=Path, default=None, help="default teachers/<task>_<k>demo")
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if (args.n_demos is None) == (args.episode_ids is None):
        raise SystemExit("give exactly one of --n-demos / --episode-ids")
    horizon = HORIZON[args.task]
    src = (args.src or PROJECT_ROOT / "teachers" / args.task / "demonstrations.pt").resolve()
    ep, n_envs = load_episodes(src, horizon)
    n_ep = ep["obs"].shape[0]
    ids = list(range(args.n_demos)) if args.n_demos is not None else [int(i) for i in args.episode_ids.split(",")]
    if not ids or max(ids) >= n_ep or min(ids) < 0 or len(set(ids)) != len(ids):
        raise SystemExit(f"episode ids must be distinct and in [0, {n_ep}); got {ids}")
    k = len(ids)
    out_dir = (args.out or PROJECT_ROOT / "teachers" / f"{args.task}_{k}demo").resolve()
    if out_dir == src.parent:
        raise SystemExit("refusing to write into the source teacher's folder")
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = {key: val[ids].reshape(k * horizon, *val.shape[2:]) for key, val in ep.items()}
    print(f"subset: episodes {ids} -> {sub['obs'].shape[0]} rows; on-goal flag rows {int(sub['dones'].sum())}; "
          f"native return mean {sub['rewards'][:, 0].reshape(k, horizon).sum(1).mean():.1f}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, train_info = fit(sub["obs"], sub["act"], hidden=args.hidden, layers=args.layers, lr=args.lr,
                            batch=args.batch, max_epochs=args.max_epochs, patience=args.patience,
                            min_delta=args.min_delta, seed=args.seed, device=device)

    model.cpu().save_checkpoint(str(out_dir / "policy_checkpoint.pth"))
    torch.save({key: torch.from_numpy(val) for key, val in sub.items()}, out_dir / "demonstrations.pt")
    meta = dict(
        command_line=" ".join(sys.argv), timestamp=time.strftime("%Y%m%d_%H%M%S"),
        scenario=args.task, source=str(src.relative_to(PROJECT_ROOT)) if src.is_relative_to(PROJECT_ROOT) else str(src),
        source_env_interleave=int(n_envs), episode_ids=ids, episodes=k, transitions=int(sub["obs"].shape[0]),
        horizon=horizon, reward_mode="dense (raw recording values)", layout="episode-contiguous",
        model=dict(type="DecentralizedMiniBC", n=int(sub["obs"].shape[1]), in_size=int(sub["obs"].shape[1] * sub["obs"].shape[2]),
                   out_size=int(sub["act"].shape[1] * sub["act"].shape[2]), hidden_size=args.hidden, hidden_layers=args.layers),
        training=dict(lr=args.lr, batch=args.batch, max_epochs=args.max_epochs, patience=args.patience,
                      min_delta=args.min_delta, seed=args.seed, **train_info),
        note=("analysis/train_bc_subset.py: BC refit on a subset of the R2BC recording, NOT a smaller R2BC run. "
              "Relabel with analysis/relabel_demos_sparse.py --src/--dst before use under the sparse schema."),
    )
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_dir}/{{policy_checkpoint.pth, demonstrations.pt, metadata.json}}  {train_info}")


if __name__ == "__main__":
    main()
