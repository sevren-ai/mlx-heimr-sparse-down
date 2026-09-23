"""NVIDIA's multi-token-prediction head for Nemotron 3.5 Lightning, as a drafter.

The model's own release ships an MTP head (its `mtp.layers.0/1` tensors): enorm, hnorm,
eh_proj, one attention block, one MoE block and a final layernorm. It predicts the token
two positions ahead from the target's hidden state at position p and the embedding of the
token at p+1:

    x = eh_proj(concat(enorm(embed(tok_{p+1})), hnorm(h_p))) -> attention -> MoE -> norm
    logits = lm_head(x)        predicts tok_{p+2}

The embedding table and lm_head are the target model's. The attention block carries a KV
cache over the rows the target has confirmed; positions are the target's.

mlx-community's 4-bit export of the model drops the `mtp.*` tensors, so the head is
published separately, quantised to the same affine 4-bit / group 64 format
(`sevren-ai/nemotron-3.5-lightning-mtp-head-mlx-4bit`, file `mtp_head.safetensors`, 772 MB),
and this class loads that file. The head is NVIDIA's; speculative decoding with a
verify-and-rollback step is the standard technique (Leviathan et al., Chen et al.); what
this repository adds is the MLX plumbing for this model.

`h_p` is fed BEFORE the target's final layernorm (prenorm=True, the default): that is the
state the head was trained on (Megatron's decoder output) and it drafts noticeably better
than the post-norm state. prenorm=False feeds the state after norm_f instead.
"""
import json
import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import nemotron_h as nh
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache

from .resolve import resolve_head

HEAD_FILE = "mtp_head.safetensors"


class MTPHead:
    """The standalone 4-bit MTP head, wired to a loaded target model."""

    def __init__(self, target, head_dir=None, prenorm=True):
        """target: a hsd.model.NemotronModel (its embedding and lm_head are shared).
        head_dir: the head's directory; resolved from the default locations when None."""
        head_dir = resolve_head(head_dir)
        args = target.args
        self.target, self.eps, self.prenorm = target, args.layer_norm_epsilon, prenorm
        self.attn = nh.NemotronHBlock(args, "*")
        self.moe = nh.NemotronHBlock(args, "E")
        w = mx.load(os.path.join(head_dir, HEAD_FILE))
        meta = json.load(open(os.path.join(head_dir, "config.json")))
        q = meta["quantization"]
        nn.quantize(self.attn, group_size=q["group_size"], bits=q["bits"])
        nn.quantize(self.moe, group_size=q["group_size"], bits=q["bits"])
        self.attn.load_weights([(k[len("attn."):], v) for k, v in w.items() if k.startswith("attn.")], strict=True)
        self.moe.load_weights([(k[len("moe."):], v) for k, v in w.items() if k.startswith("moe.")], strict=True)
        self.enorm, self.hnorm = w["enorm"], w["hnorm"]
        self.eh_proj, self.final_norm = w["eh_proj"], w["final_norm"]
        mx.eval(self.enorm, self.hnorm, self.eh_proj, self.final_norm, self.attn.parameters(), self.moe.parameters())
        del w
        self.bytes = sum(a.nbytes for a in (self.enorm, self.hnorm, self.eh_proj, self.final_norm)) + \
            sum(a.nbytes for m in (self.attn, self.moe) for _, a in tree_flatten(m.parameters()))
        self.reset()

    def reset(self):
        self.kv = KVCache()

    def trim(self, n):
        """Drop the last n rows of the head's cache (the speculative rows after a verify)."""
        self.kv.trim(n)

    def __call__(self, h, toks):
        """h [R, D]: the target's hidden state at R confirmed positions (see prenorm);
        toks [R] int32: the token that FOLLOWS each row's position (tok_{p+1} for row p)
        -> (pred [R] int32 = the argmax, i.e. the token two positions ahead of each row,
             hn [R, D] = the head's own normed hidden state, for chaining).
        """
        R = h.shape[0]
        e = mx.fast.rms_norm(self.target.embed(toks), self.enorm, self.eps)
        hh = mx.fast.rms_norm(h, self.hnorm, self.eps)
        x = (mx.concatenate([e, hh], axis=-1) @ self.eh_proj.T)[None]
        x = self.attn(x, mask=create_attention_mask(x, self.kv) if R > 1 else None, cache=self.kv)
        x = self.moe(x)
        hn = mx.fast.rms_norm(x[0], self.final_norm, self.eps)
        logits = self.target.head_fn(hn).astype(mx.float32)
        return mx.argmax(logits, axis=-1).astype(mx.int32), hn
