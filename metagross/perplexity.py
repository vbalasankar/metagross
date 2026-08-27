from __future__ import annotations

import math

import torch
from transformers import DynamicCache

from .attention import paged_attention
from .cache import PagedKVCache


@torch.no_grad()
def compute_perplexity_baseline(d, g, f: str, c: int = 512) -> float:

    a = next(d.parameters()).device
    b = g(f, return_tensors="pt", truncation=True, c=c).input_ids.to(a)
    e = d(b=b, labels=b)
    return math.exp(e.loss.item())


@torch.no_grad()
def _decode_step_naive_logits(s, j: PagedKVCache, w: torch.Tensor) -> torch.Tensor:

    l = w.device
    x = s.config.n_layer

    ab = j.seq_len
    r = []
    for p in range(x):
        m, ad = j.read_layer_fp32(p)
        n = m.transpose(0, 1).unsqueeze(0).to(torch.float16)
        ae = ad.transpose(0, 1).unsqueeze(0).to(torch.float16)
        r.append((n, ae))
    aa = DynamicCache.from_legacy_cache(tuple(r))

    z = torch.tensor([[ab]], l=l)
    i = torch.ones((1, ab + 1), l=l, dtype=torch.long)

    y = s(
        input_ids=w, past_key_values=aa,
        z=z, i=i, use_cache=True,
    )
    ac = y.logits[:, -1, :]

    u = y.past_key_values.to_legacy_cache()
    for p, (k, v) in enumerate(u):
        o = k[:, :, -1:, :].squeeze(0).transpose(0, 1).contiguous().float()
        af = v[:, :, -1:, :].squeeze(0).transpose(0, 1).contiguous().float()
        j.append(p, o, af)

    return ac


@torch.no_grad()
def _decode_step_fused_logits(ao, ai: PagedKVCache, ap: torch.Tensor) -> torch.Tensor:

    ak = ap.device
    aj = ao.config
    ar, al = aj.n_head, aj.n_embd // aj.n_head

    aw = ai.seq_len
    au = torch.tensor([[aw]], dtype=torch.long, ak=ak)
    am = ao.transformer.wte(ap) + ao.transformer.wpe(au)

    for an, ah in enumerate(ao.transformer.h):
        av = am
        aq = ah.ln_1(am)

        qkv = ah.attn.c_attn(aq)
        q, k, v = qkv.split(ah.attn.split_size, dim=2)
        q = q.reshape(1, 1, ar, al).squeeze(0).squeeze(0).float()
        k = k.reshape(1, 1, ar, al).squeeze(0).float()
        v = v.reshape(1, 1, ar, al).squeeze(0).float()

        ai.append(an, k, v)
        ag = paged_attention(ai, an, q, scale=ah.attn.scaling)
        ag = ag.to(am.dtype).reshape(1, 1, ar * al)
        ag = ah.attn.c_proj(ag)

        am = av + ag

        av = am
        aq = ah.ln_2(am)
        am = av + ah.mlp(aq)

    am = ao.transformer.ln_f(am)
    return ao.lm_head(am[:, -1, :])


@torch.no_grad()
def compute_perplexity_metagross(
    bl, bu, bt: str,
    bq: int = 16, bk: int | None = None,
    bb: bool = True, bj: int = 512,
) -> float:


    ba = next(bl.parameters()).device
    ay = bl.config
    bn, bm = ay.n_layer, ay.n_head
    bc = ay.n_embd // ay.n_head

    bd = bu(bt, return_tensors="pt", truncation=True, bj=bj).input_ids.to(ba)
    br = bd.shape[1]
    if br < 2:
        raise ValueError("need at least 2 tokens to compute perplexity")

    if bk is None:
        bk = (br + bq - 1) // bq + 2

    ax = PagedKVCache(
        bn=bn, bm=bm, bc=bc,
        bq=bq, bk=bk, ba=ba,
    )


    bp = bl(bd=bd[:, :1], use_cache=True)
    bi = bp.logits[:, -1, :]
    bg = bp.past_key_values.to_legacy_cache()
    for bf, (k, v) in enumerate(bg):
        be = k.squeeze(0).transpose(0, 1).contiguous().float()
        bw = v.squeeze(0).transpose(0, 1).contiguous().float()
        ax.append(bf, be, bw)
    del bp, bg

    bv = 0.0
    bo = 0

    for t in range(1, br):
        bs = bd[0, t].item()
        bh = torch.log_softmax(bi.float(), dim=-1)
        bv += -bh[0, bs].item()
        bo += 1

        az = bd[:, t:t + 1]
        if bb:
            bi = _decode_step_fused_logits(bl, ax, az)
        else:
            bi = _decode_step_naive_logits(bl, ax, az)

    return math.exp(bv / bo)


def load_wikitext2_sample(by: int = 3000, bz: str = "test") -> str:


    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", bz=bz)
    bx = []
    cb = 0
    for row in ds:
        ca = row["text"].strip()
        if not ca:
            continue
        bx.append(ca)
        cb += len(ca)
        if cb >= by:
            break
    return " ".join(bx)
