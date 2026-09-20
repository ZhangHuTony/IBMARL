"""
Repository-root-anchored path resolution.

Artifact paths in the configs (``r2bc_checkpoint_path``, ``demonstrations_path``)
are written relative to the repository root -- ``teachers/navigation/...`` -- so
a checkout runs on any machine.  Every documented entry point already assumes
CWD == repo root, but the archived ``results/<run>/config.yaml`` snapshots are
re-read by analysis tooling from arbitrary directories, so resolution is
anchored on this file's location rather than on the working directory.

Absolute paths pass through untouched.  That is how transport's 91 MB
demonstration file, too large to bundle, stays a documented machine-local
dependency (see teachers/README.md).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path) -> Path:
    """*path* as given if absolute, else relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
