from __future__ import annotations

import math

import torch
from transformers import DynamicCache

from .attention import paged_attention
from .cache import PagedKVCache


@torch.no_grad()
def compute_perplexity_baseline(model, tokenizer, text, max_length = 512) -> float:

    a = next(model.parameters()).device
    b = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(a)
    e = model(input_ids=b, labels=b)
    return math.exp(e.loss.item())


@torch.no_grad()
def _decode_step_naive_logits(model, cache, next_token) -> torch.Tensor:

    l = next_token.device
    x = model.config.n_layer

    ab = cache.seq_len
    r = []
    for p in range(x):
        m, ad = cache.read_layer_fp32(p)
        n = m.transpose(0, 1).unsqueeze(0).to(torch.float16)
        ae = ad.transpose(0, 1).unsqueeze(0).to(torch.float16)
        r.append((n, ae))
    aa = DynamicCache.from_legacy_cache(tuple(r))

    z = torch.tensor([[ab]], device=l)
    i = torch.ones((1, ab + 1), device=l, dtype=torch.long)

    y = model(
        input_ids=next_token, past_key_values=aa,
        position_ids=z, attention_mask=i, use_cache=True,
    )
    ac = y.logits[:, -1, :]

    u = y.past_key_values.to_legacy_cache()
    for p, (k, v) in enumerate(u):
        o = k[:, :, -1:, :].squeeze(0).transpose(0, 1).contiguous().float()
        af = v[:, :, -1:, :].squeeze(0).transpose(0, 1).contiguous().float()
        cache.append(p, o, af)

    return ac


@torch.no_grad()
def _decode_step_fused_logits(model, cache, next_token) -> torch.Tensor:

    ak = next_token.device
    aj = model.config
    ar, al = aj.n_head, aj.n_embd // aj.n_head

    aw = cache.seq_len
    au = torch.tensor([[aw]], dtype=torch.long, device=ak)
    am = model.transformer.wte(next_token) + model.transformer.wpe(au)

    for an, ah in enumerate(model.transformer.h):
        av = am
        aq = ah.ln_1(am)

        qkv = ah.attn.c_attn(aq)
        q, k, v = qkv.split(ah.attn.split_size, dim=2)
        q = q.reshape(1, 1, ar, al).squeeze(0).squeeze(0).float()
        k = k.reshape(1, 1, ar, al).squeeze(0).float()
        v = v.reshape(1, 1, ar, al).squeeze(0).float()

        cache.append(an, k, v)
        ag = paged_attention(cache, an, q, scale=ah.attn.scaling)
        ag = ag.to(am.dtype).reshape(1, 1, ar * al)
        ag = ah.attn.c_proj(ag)

        am = av + ag

        av = am
        aq = ah.ln_2(am)
        am = av + ah.mlp(aq)

    am = model.transformer.ln_f(am)
    return model.lm_head(am[:, -1, :])


@torch.no_grad()
def compute_perplexity_metagross(
    model, tokenizer, text,
    page_size = 16, max_pages = None,
    fused = True, max_length = 512,
) -> float:


    ba = next(model.parameters()).device
    ay = model.config
    bn, bm = ay.n_layer, ay.n_head
    bc = ay.n_embd // ay.n_head

    bd = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(ba)
    br = bd.shape[1]
    if br < 2:
        raise ValueError("need at least 2 tokens to compute perplexity")

    if max_pages is None:
        max_pages = (br + page_size - 1) // page_size + 2

    ax = PagedKVCache(
        num_layers=bn, num_heads=bm, head_dim=bc,
        page_size=page_size, max_pages=max_pages, device=ba,
    )


    bp = model(input_ids=bd[:, :1], use_cache=True)
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
        if fused:
            bi = _decode_step_fused_logits(model, ax, az)
        else:
            bi = _decode_step_naive_logits(model, ax, az)

    return math.exp(bv / bo)


def load_wikitext2_sample(num_chars = 3000, split = "test") -> str:


    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    bx = []
    cb = 0
    for row in ds:
        ca = row["text"].strip()
        if not ca:
            continue
        bx.append(ca)
        cb += len(ca)
        if cb >= num_chars:
            break
    return " ".join(bx)
