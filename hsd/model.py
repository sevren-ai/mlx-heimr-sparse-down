"""Nemotron 3.5 Lightning 30B A3B end to end on MLX, from the stock 4-bit checkpoint.

The model itself is mlx-lm's own (`mlx_lm.load`), so dense mode is block-for-block what
`mlx_lm.generate` runs and is the baseline in every comparison. What this module adds:

  sparse   the MoE down projection runs over the live rows only, on a neuron-major
           permutation of the checkpoint's own bytes built once at load (hsd.layout,
           hsd.decode). The checkpoint's down layout is then freed, so exactly one layout
           is resident and the model needs no more memory than dense mode. Prefill runs
           on the same layout with the matrix-unit kernel (hsd.prefill), because the
           permuted weights are transposed relative to what MLX's dense quantised GEMM
           consumes and keeping both layouts would cost 8.4 GB.

  mtp      greedy speculative decoding with NVIDIA's own MTP head (hsd.mtp): per step the
           head drafts k tokens, the target verifies anchor + drafts in one step of k+1
           rows, the attention KV caches are trimmed and the Mamba states recomputed to
           the accepted prefix (hsd.mamba).

Generation loops are pipelined: the next step is built and submitted before the current
token is read back, and the wait is `y.item()`.
"""
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm import load as mlx_load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

from .decode import sparse_down
from .layout import to_neuron_major
from .mamba import mamba_rollback, mamba_verify
from .prefill import build_tiles, build_union, sparse_down_prefill
from .resolve import resolve_model

MODES = ("dense", "sparse")

# Measured defaults (see the benchmarks in bench/).
DECODE_KW = dict(sl=928, unr=2, nds=3)        # decode kernel: slice, unroll, output chunks
PREFILL_KW = dict(tt=32, nb=64, kc=32, nsg=4)  # prefill kernel: tile, block, K chunk, simdgroups
# Layers whose per-tile union of live neurons is dense enough that a 64-slot tile beats a
# 32-slot one: the B staging is paid once per tile and the union barely grows.
PREFILL_TT64_LAYERS = (17, 20, 22, 24, 27, 29, 31, 51)


def relu2(x):
    return mx.square(nn.relu(x))


class MoEBlock:
    """One mlx-lm MoE block wired for the two down-projection modes.

    dense:   the stock block, unchanged.
    sparse:  the stock router and up projection, ReLU2, then the down projection over the
             live rows on the neuron-major layout (decode) or over the per-tile union of
             live neurons on the matrix units (prefill).
    """

    def __init__(self, block, args, decode_kw=DECODE_KW, prefill_kw=PREFILL_KW):
        self.block = block
        mix = block.mixer
        self.topk = args.num_experts_per_tok
        self.F = args.moe_intermediate_size
        self.FS = args.moe_shared_expert_intermediate_size
        self.D = args.hidden_size
        self.E = args.n_routed_experts
        self.group_size = mix.switch_mlp.fc1.group_size
        self.bits = mix.switch_mlp.fc1.bits
        self.decode_kw = dict(decode_kw)
        self.prefill_kw = dict(prefill_kw)
        self.has_layout = False      # the neuron-major layout has been built
        self.stock_layout = True     # the checkpoint's own down layout is still resident
        self.mode = "dense"

    # ---- the neuron-major layout
    @property
    def w_routed(self):
        """(pw, ps, pb): the routed experts' down projection, neuron-major."""
        return (self.pw, self.ps, self.pb)

    @property
    def w_shared(self):
        """(spw, sps, spb): the shared expert's down projection, neuron-major."""
        return (self.spw, self.sps, self.spb)

    def build_layout(self, drop_stock=True):
        """Permute this block's down projections into the neuron-major layout (an exact
        permutation of the checkpoint's bytes, hsd.layout); with drop_stock, free the
        checkpoint's own layout so only one copy is resident."""
        mix = self.block.mixer
        fc2, shared = mix.switch_mlp.fc2, mix.shared_experts.down_proj
        self.pw, self.ps, self.pb = to_neuron_major(fc2.weight, fc2.scales, fc2.biases)
        self.spw, self.sps, self.spb = to_neuron_major(shared.weight, shared.scales, shared.biases)
        mx.eval(self.pw, self.ps, self.pb, self.spw, self.sps, self.spb)
        self.has_layout = True
        if drop_stock:
            self.drop_stock()

    def drop_stock(self):
        mix = self.block.mixer
        for mod in (mix.switch_mlp.fc2, mix.shared_experts.down_proj):
            for k in ("weight", "scales", "biases"):
                mod[k] = mx.zeros((1,), dtype=mod[k].dtype)
        self.stock_layout = False

    # ---- shared front end: router, up projections, ReLU2
    def front(self, x):
        """x [B, D] -> (inds [B, topk], scores [B, topk], h [B, topk, 1, F], hs [B, FS]),
        with h / hs after ReLU2."""
        block, mix = self.block, self.block.mixer
        xn = block.norm(x)
        inds, scores = mix.gate(xn)
        fc1 = mix.switch_mlp.fc1
        u = mx.gather_qmm(xn[:, None, None, :], fc1.weight, fc1.scales, fc1.biases,
                          rhs_indices=inds, transpose=True, group_size=self.group_size, bits=self.bits)
        h = relu2(u)
        hs = relu2(mix.shared_experts.up_proj(xn))
        return inds, scores, h, hs

    def pack(self, inds, scores, h, hs):
        """The down kernel's inputs: h [B, topk*F + FS] (the routed experts then the shared
        one, contiguous), the selected experts, and the gains (routing scores, 1 for shared)."""
        B = h.shape[0]
        hcat = mx.concatenate([h.reshape(B, self.topk * self.F), hs], axis=1)
        gains = mx.concatenate([scores.astype(mx.float32), mx.ones((B, 1))], axis=1)
        return hcat, inds.astype(mx.int32), gains

    # ---- decode (x [B, D], B = 1 for plain steps, B = k+1 for a verify step)
    def decode_dense(self, x):
        return self.block(x[None])[0]

    def decode_sparse(self, x):
        hcat, sel, gains = self.pack(*self.front(x))
        y = sparse_down(hcat, sel, gains, self.pw, self.ps, self.pb, self.spw, self.sps, self.spb,
                        F=self.F, FS=self.FS, D=self.D, topk=self.topk, **self.decode_kw)
        return x + y.astype(x.dtype)

    def decode_fn(self, mode):
        """The compiled decode function for this mode (mx.compile traces per row count, so
        the multi-row verify step gets its own trace)."""
        self.mode = mode
        if mode == "dense":
            return mx.compile(self.decode_dense)
        assert self.has_layout, "sparse mode needs the neuron-major layout (build_layout)"
        return mx.compile(self.decode_sparse)

    # ---- prefill (x [T, D])
    def prefill(self, x):
        if self.mode == "sparse":
            return self.prefill_sparse(x)
        return self.block(x[None])[0]

    def prefill_sparse(self, x):
        """The stock sorted up projection, then the down projection over the per-tile union
        of live neurons on the matrix-unit kernel. Tile order, union lists and gather all
        on the GPU."""
        block, mix = self.block, self.block.mixer
        xn = block.norm(x)
        inds, scores = mix.gate(xn)
        xs, idx, inv = _gather_sort(xn[:, None, None, :], inds)
        h = relu2(mix.switch_mlp.fc1(xs, idx, sorted_indices=True))
        h = _scatter_unsort(h, inv, inds.shape).squeeze(-2)          # [T, topk, F], token order
        hs = relu2(mix.shared_experts.up_proj(xn))
        hcat, sel, gains = self.pack(inds, scores, h, hs)
        kw = self.prefill_kw
        tiles = build_tiles(sel, self.E, kw["tt"])
        union = build_union(hcat, tiles[0], tiles[1], F=self.F, FS=self.FS, topk=self.topk,
                            E=self.E, tt=kw["tt"], kc=kw["kc"])
        y = sparse_down_prefill(hcat, sel, gains, self.pw, self.ps, self.pb, self.spw, self.sps, self.spb,
                                F=self.F, FS=self.FS, D=self.D, topk=self.topk, E=self.E,
                                tiles=tiles, union=union, **kw)
        return x + y.astype(x.dtype)


class NemotronModel:
    """The whole model: mlx-lm's own blocks with our MoE down-projection modes and our
    generation loops on top of them."""

    def __init__(self, model_path=None, mode="sparse", verbose=True, wire=True,
                 single_layout=True, n_layers=None):
        """model_path: a local directory or an HF repo id (hsd.resolve).
        mode: 'dense' (stock blocks) or 'sparse' (the neuron-major down projection).
        single_layout: in sparse mode, free the checkpoint's down layout layer by layer as
            the neuron-major one is built, so only one layout is ever resident (prefill then
            necessarily runs on the matrix-unit kernel). The model then needs no more
            memory than dense mode. Set False to keep both (benchmarks that compare them).
        n_layers: debug, load only the first N layers."""
        self.path = resolve_model(model_path)
        if wire:
            try:
                mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
            except Exception:
                pass
        t0 = time.time()
        self.model, self.tokenizer = mlx_load(self.path, tokenizer_config={}, lazy=False)
        self.args = self.model.args
        chars = self.args.hybrid_override_pattern[:n_layers] if n_layers else self.args.hybrid_override_pattern
        self.types = [{"M": "mamba", "*": "attention", "E": "moe"}[c] for c in chars]
        self.D = self.args.hidden_size
        self.eps = self.args.layer_norm_epsilon
        bb = self.model.backbone
        self.embeddings, self.norm_f, self.lm_head = bb.embeddings, bb.norm_f, self.model.lm_head
        self.layers = bb.layers[: len(self.types)]
        self.single_layout = single_layout
        self.blocks = []
        for i, t in enumerate(self.types):
            if t == "moe":
                kw = dict(PREFILL_KW, tt=64 if i in PREFILL_TT64_LAYERS else PREFILL_KW["tt"])
                self.blocks.append(MoEBlock(self.layers[i], self.args, prefill_kw=kw))
            else:
                self.blocks.append(self.layers[i])
        self.moe_layers = [l for l, t in enumerate(self.types) if t == "moe"]
        if verbose:
            print(f"loaded {len(self.types)} layers  ({mx.get_active_memory() / 1e9:.1f} GB active, {time.time() - t0:.0f}s)", flush=True)
        self.mode = None
        self.set_mode(mode)
        self.load_seconds = time.time() - t0

    # ---- pieces shared with the MTP head
    def embed(self, ids):
        return self.embeddings(ids)

    def head_fn(self, h):
        return self.lm_head(h)

    def logits(self, x):
        return self.lm_head(self.norm_f(x))

    # ---- modes
    def set_mode(self, mode):
        assert mode in MODES, mode
        if mode == "sparse":
            t0 = time.time()
            for l in self.moe_layers:
                blk = self.blocks[l]
                if not blk.has_layout:
                    # one layer at a time: the transient of the permutation stays bounded
                    # and the freed bytes are reusable for the next layer
                    blk.build_layout(drop_stock=self.single_layout)
                    mx.clear_cache()
            if time.time() - t0 > 0.5:
                print(f"neuron-major down layouts built in {time.time() - t0:.1f}s "
                      f"({mx.get_active_memory() / 1e9:.1f} GB active)", flush=True)
        else:
            for l in self.moe_layers:
                assert self.blocks[l].stock_layout, "dense mode needs the checkpoint's down layout, freed by sparse mode"
        self.moe_fn = {l: self.blocks[l].decode_fn(mode) for l in self.moe_layers}
        self.mode = mode
        mx.clear_cache()

    def make_cache(self):
        return [ArraysCache(size=2) if t == "mamba" else KVCache() if t == "attention" else None
                for t in self.types]

    # ---- forward
    def prefill(self, tokens, cache, chunk=2048, on_hidden=None):
        """tokens [T] -> the residual stream of the last position [1, D].
        on_hidden(start, x [t, D]) is called per chunk with the chunk's residual stream
        BEFORE norm_f (the MTP head's prefill hook)."""
        T = tokens.shape[0]
        last = None
        for i in range(0, T, chunk):
            x = self.embed(tokens[i:i + chunk])
            for l, bt in enumerate(self.types):
                blk, c = self.blocks[l], cache[l]
                if bt == "moe":
                    x = blk.prefill(x)
                else:
                    x3 = x[None]
                    mask = create_attention_mask(x3, c) if bt == "attention" else None
                    x = blk(x3, mask=mask, cache=c)[0]
            if on_hidden is not None:
                on_hidden(i, x)
            last = x[-1:]
            mx.eval(last, *[s for c in cache if c is not None for s in c.state])
        return last

    def step(self, x):
        """One decode step on x [1, D], through every block, with self._cache."""
        for l, bt in enumerate(self.types):
            if bt == "moe":
                x = self.moe_fn[l](x)
            else:
                x = self.blocks[l](x[:, None, :], mask=None, cache=self._cache[l])[:, 0]
        return x

    # ---- plain generation
    def generate(self, prompt, max_tokens=128, temp=0.0, eos=(), chunk=2048, cache=None):
        """Yields (token, stats) per token; stats is None except on the last yield.
        Greedy (temp=0) or sampled. The step after the current token is built and
        submitted before the current token is read back; the wait is `y.item()` -- an
        `int(y[0])` would enqueue an op behind the submitted step and destroy the overlap."""
        prompt = mx.array(prompt, dtype=mx.int32)
        self._cache = cache or self.make_cache()
        t0 = time.perf_counter()
        h = self.prefill(prompt, self._cache, chunk)

        def sample(h):
            lg = self.logits(h).astype(mx.float32)
            return mx.argmax(lg, axis=-1) if temp == 0 else mx.random.categorical(lg / temp)

        y = sample(h)
        mx.eval(y)
        prompt_s = time.perf_counter() - t0
        tok = int(y.item())
        out = [tok]
        t1 = time.perf_counter()
        n = 1
        y_next = sample(self.step(self.embed(y).reshape(1, self.D)))
        mx.async_eval(y_next)
        while n < max_tokens and tok not in eos:
            yield tok, None
            y_after = sample(self.step(self.embed(y_next).reshape(1, self.D)))
            mx.async_eval(y_after)
            tok = int(y_next.item())
            out.append(tok)
            n += 1
            y_next = y_after
        dec_s = time.perf_counter() - t1
        yield tok, self._stats(prompt, prompt_s, n, dec_s, out, temp=temp)

    def _stats(self, prompt, prompt_s, n, dec_s, out, **extra):
        return dict(mode=self.mode, prompt_tokens=int(prompt.shape[0]), prompt_s=prompt_s,
                    prompt_tps=prompt.shape[0] / prompt_s, gen_tokens=n, decode_s=dec_s,
                    decode_tps=(n - 1) / dec_s if n > 1 else float("nan"),
                    ms_per_token=1e3 * dec_s / max(n - 1, 1),
                    peak_gb=mx.get_peak_memory() / 1e9, tokens=out, **extra)

    # ---- the verify step and the rollback shared by speculative decoding
    def verify_step(self, x, cache):
        """x [B, D]: B consecutive positions through every layer in ONE step, every cache
        advanced by B rows. Returns the final residual stream [B, D]. Remembers what
        rollback() needs (the Mamba recurrence inputs of these rows)."""
        self._records = []
        for l, bt in enumerate(self.types):
            if bt == "moe":
                x = self.moe_fn[l](x)
            elif bt == "attention":
                x3 = x[None]
                x = self.blocks[l](x3, mask=create_attention_mask(x3, cache[l]), cache=cache[l])[0]
            else:
                x = mamba_verify(self.blocks[l], x[None], cache[l], self._records)[0]
        return x

    def rollback(self, cache, n_keep, B):
        """Undo a verify_step of B rows beyond its first n_keep: attention caches are
        trimmed, Mamba states recomputed from the recorded inputs over the kept prefix."""
        if n_keep == B:
            return
        for l, bt in enumerate(self.types):
            if bt == "attention":
                cache[l].trim(B - n_keep)
        mamba_rollback(self._records, n_keep)

    @staticmethod
    def _spec_stats(n_draft, steps, accepted, dec_s, n):
        return dict(n_draft=n_draft, steps=steps, ms_per_step=1e3 * dec_s / max(steps, 1),
                    tokens_per_step=(n - 1) / max(steps, 1),
                    accepted_mean=float(np.mean(accepted)) if accepted else 0.0,
                    accepted_hist=[int(sum(1 for v in accepted if v == k)) for k in range(n_draft + 1)])

    # ---- MTP speculative decoding
    def generate_mtp(self, prompt, mtp, n_draft=2, max_tokens=128, eos=(), chunk=2048):
        """Greedy speculative decoding with the MTP head. Yields (token, stats); stats is
        None except on the last yield.

        Per step: the head first runs over the rows the previous verify confirmed (target
        hidden at position p, token at p+1 -> its prediction of the token at p+2); the last
        row's prediction is the draft, and with n_draft > 1 the head is chained on its own
        hidden output for the further drafts. Those chained rows leave the head's cache
        again after the verify (it only ever holds confirmed rows). The target verifies
        [anchor, drafts] in one step of n_draft+1 rows; with greedy sampling the accepted
        prefix is the longest run of drafts equal to the target's own argmax, and the
        target's argmax at the first mismatch is the bonus token. Greedy only."""
        assert n_draft >= 1
        prompt = mx.array(prompt, dtype=mx.int32)
        cache = self.make_cache()
        mtp.reset()
        hidden = (lambda x: x) if mtp.prenorm else self.norm_f
        last_h = []

        def on_hidden(i, x):                       # the head's prefill: rows (h_p, tok_{p+1}), p < T-1
            hp = hidden(x)
            nxt = prompt[i + 1: i + 1 + x.shape[0]]
            if nxt.shape[0] > 0:
                mtp(hp[: nxt.shape[0]], nxt)
            mx.eval(*mtp.kv.state)                 # do not hold the whole prefill graph
            last_h.append(hp[-1:])

        t0 = time.perf_counter()
        x_last = self.prefill(prompt, cache, chunk, on_hidden=on_hidden)
        y = mx.argmax(self.logits(x_last).astype(mx.float32), axis=-1)
        mx.eval(y)
        prompt_s = time.perf_counter() - t0
        tok = int(y.item())
        out = [tok]
        pend_h, pend_t = [last_h[-1]], [tok]       # rows the head has not seen yet: (hidden at p, token at p+1)
        pending = tok
        t1 = time.perf_counter()
        n = 1
        steps = 0
        acc = []
        bs = n_draft + 1
        while n < max_tokens and tok not in eos:
            d, hd = mtp(mx.concatenate(pend_h, axis=0), mx.array(pend_t, dtype=mx.int32))
            drafts = [d[-1:]]
            hd = hd[-1:]
            for _ in range(1, n_draft):            # chain the head on its own hidden output
                d, hd = mtp(hd, drafts[-1])
                drafts.append(d)
            draft = mx.concatenate(drafts) if n_draft > 1 else drafts[0]
            mx.async_eval(draft)
            ids = mx.concatenate([mx.array([tok], dtype=mx.int32), draft])
            x = self.verify_step(self.embed(ids), cache)
            hv = hidden(x)
            tgt = mx.argmax(self.head_fn(hv if not mtp.prenorm else self.norm_f(x)).astype(mx.float32), axis=-1)
            mx.async_eval(tgt, hv)
            dl = draft.tolist()
            tl = tgt.tolist()
            a = next((i for i in range(bs - 1) if dl[i] != tl[i]), bs - 1)
            new = (dl[:a] + [tl[a]])[: max_tokens - n]
            self.rollback(cache, a + 1, bs)
            if n_draft > 1:
                mtp.trim(n_draft - 1)              # drop the chained speculative rows
            pend_h = [hv[i:i + 1] for i in range(len(new))]
            pend_t = list(new)
            steps += 1
            acc.append(a)
            for tk in new:
                yield pending, None
                pending = tk
                out.append(tk)
                n += 1
                tok = tk
                if tk in eos:
                    break
        dec_s = time.perf_counter() - t1
        yield pending, self._stats(prompt, prompt_s, n, dec_s, out, temp=0.0,
                                   spec=dict(kind="mtp", prenorm=mtp.prenorm,
                                             **self._spec_stats(n_draft, steps, acc, dec_s, n)))
