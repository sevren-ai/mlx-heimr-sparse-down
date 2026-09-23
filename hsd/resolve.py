"""Finding the model and the MTP head on disk.

For an HF repo id `<org>/<name>`, in order:

  1. a local path always wins (the argument is an existing directory)
  2. LM Studio's model directories: `~/.lmstudio/models/<org>/<name>`, then
     `~/.cache/lm-studio/models/<org>/<name>`
  3. the Hugging Face cache
  4. `huggingface_hub.snapshot_download`

so a machine that already has the 17.7 GB checkpoint in LM Studio does not download it
again, and a machine without it gets it with no extra steps.
"""
import os

from huggingface_hub import snapshot_download

DEFAULT_MODEL_ID = "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
DEFAULT_MTP_HEAD_ID = "sevren-ai/nemotron-3.5-lightning-mtp-head-mlx-4bit"


def resolve_repo(id_or_path, default_id=None, download=True):
    """-> a local directory holding the model / head, or None when nothing is on disk and
    download is False. `id_or_path` may be None (use the default id), a local directory,
    or an HF repo id."""
    repo = id_or_path or default_id
    if os.path.isdir(repo):
        return repo
    parts = repo.split("/")
    if len(parts) != 2:
        raise FileNotFoundError(f"{repo!r} is neither an existing directory nor an HF repo id org/name")
    org, name = parts
    home = os.path.expanduser("~")
    for d in (os.path.join(home, ".lmstudio", "models", org, name),
              os.path.join(home, ".cache", "lm-studio", "models", org, name)):
        if os.path.isdir(d):
            return d
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        pass
    if not download:
        return None
    print(f"downloading {repo} ...", flush=True)
    return snapshot_download(repo)


def resolve_model(id_or_path=None, download=True):
    """The 4-bit checkpoint (default: mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit)."""
    return resolve_repo(id_or_path, DEFAULT_MODEL_ID, download)


def resolve_head(id_or_path=None, download=True):
    """The standalone MTP head (default: sevren-ai/nemotron-3.5-lightning-mtp-head-mlx-4bit,
    772 MB: mtp_head.safetensors plus its config)."""
    return resolve_repo(id_or_path, DEFAULT_MTP_HEAD_ID, download)
