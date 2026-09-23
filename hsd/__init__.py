"""Sparse down projection and MTP speculative decoding for NVIDIA's
Nemotron 3.5 Lightning 30B A3B on MLX, from the stock mlx-community 4-bit checkpoint.

Modules:
    layout    the neuron-major permutation of the checkpoint's down projections
    decode    the decode Metal kernel over the live rows
    prefill   the prefill matrix-unit kernel (tile order, union, gather, GEMM)
    mamba     multi-row Mamba-2 steps with rollback, for speculative decoding
    mtp       NVIDIA's multi-token-prediction head as the drafter
    model     the model wrapper and the generation loops (plain and MTP)
    resolve   model / head resolution (LM Studio, HF cache, download)
    cli       the command-line entry point (python -m hsd)
"""

__version__ = "0.1.0"
