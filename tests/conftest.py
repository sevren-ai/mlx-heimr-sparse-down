"""Shared fixtures for the test suite. The tests need the real 17.7 GB checkpoint (and the
MTP head for the end-to-end speculative test); when they are not on disk they are skipped
with a message rather than downloading."""
import os

import mlx.core as mx
import pytest

from hsd.resolve import resolve_head, resolve_model


def _model_dir():
    return resolve_model(download=False)


def _head_dir():
    return resolve_head(download=False)


@pytest.fixture(scope="session")
def model_dir():
    d = _model_dir()
    if d is None:
        pytest.skip("the 4-bit checkpoint is not on disk; get it with "
                    "`hf download mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit` "
                    "(or point --model at an LM Studio copy)")
    return d


@pytest.fixture(scope="session")
def head_dir():
    d = _head_dir()
    if d is None:
        pytest.skip("the MTP head is not on disk; get it with "
                    "`hf download sevren-ai/nemotron-3.5-lightning-mtp-head-mlx-4bit`")
    return d


@pytest.fixture(scope="session")
def wired():
    """Set the wired memory limit to the device's recommended working set, as the model
    wrapper does; ignore machines / versions where that is not allowed."""
    try:
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        pass
