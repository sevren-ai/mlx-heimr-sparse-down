"""The run-record metadata every benchmark JSON and the CLI record carry: the device, the
MLX version, this repository's short commit hash (blank when unavailable) and a UTC
timestamp. No model needed."""
import re
import subprocess

from hsd.runtime import git_commit, record_meta, utc_now


def test_record_meta_fields():
    m = record_meta()
    assert set(m) == {"device", "memory_gb", "mlx", "git", "utc"}
    assert m["device"]           # a name, e.g. "Apple M5 Pro"
    assert m["memory_gb"] > 0
    assert re.match(r"^\d+\.\d+\.\d+", m["mlx"])
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}Z$", m["utc"])


def test_utc_now_is_utc():
    assert utc_now().endswith("Z")


def test_git_commit_matches_repo():
    """Inside this repository the hash matches git rev-parse; the helper never raises."""
    c = git_commit()
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except Exception:
        return                          # no git available; the helper must still return a string
    assert c == out.split()[0] if out else c == ""


def main():
    import sys
    sys.exit(__import__("pytest").main([__file__, "-v", "-s"]))


if __name__ == "__main__":
    main()
