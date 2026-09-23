"""Runtime facts for run records: the device, the MLX version, this repository's git
commit, and a UTC timestamp. Shared by the CLI and the benchmark scripts so every JSON
record carries the same fields."""
import os
import subprocess
from datetime import datetime, timezone

import mlx.core as mx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_commit():
    """This repository's short commit hash (git rev-parse), blank outside a repo or
    without git. Runs against the repository this file is installed from, so it works
    from any working directory."""
    try:
        out = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out.split()[0] if out else ""
    except Exception:
        return ""


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def record_meta():
    """The common record fields: device name, its memory, the MLX version, the repo
    commit (blank when unavailable) and a UTC timestamp."""
    dev = mx.device_info()
    return dict(device=dev.get("device_name"), memory_gb=dev.get("memory_size", 0) / 1e9,
                mlx=mx.__version__, git=git_commit(), utc=utc_now())
