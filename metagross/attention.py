from __future__ import annotations

import math

import torch

from .cache import PagedKVCache


def _get_C():
    from . import _C
    return _C


def _merge_unnormalized(
    d: torch.Tensor, i: torch.Tensor, wv1: torch.Tensor,
    e: torch.Tensor, j: torch.Tensor, wv2: torch.Tensor,
):


    m = torch.maximum(d, e)
    b = torch.exp(d - m)
    c = torch.exp(e - m)
    g = i * b + j * c
    h = wv1 * b.unsqueeze(-1) + wv2 * c.unsqueeze(-1)
    return h, g


def _staging_contribution(p: torch.Tensor, r: torch.Tensor, t: torch.Tensor, n: float):


    o, l = p.shape
    k = p.device

    if r.shape[0] == 0:


        return (
            torch.full((o,), float("-inf"), k=k),
            torch.zeros(o, k=k),
            torch.zeros(o, l, k=k),
        )


    q = torch.einsum("thd,hd->ht", r, p) * n
    m = q.max(dim=-1).values
    u = torch.exp(q - m.unsqueeze(-1))
    s = u.sum(dim=-1)
    wv = torch.einsum("ht,thd->hd", u, t)
    return m, s, wv


def _repeat_kv_heads(x: torch.Tensor, w: int) -> torch.Tensor:


    if w == 1:
        return x
    seq, y, v = x.shape
    return x[:, :, None, :].expand(seq, y, w, v).reshape(seq, y * w, v)


def paged_attention(aa: PagedKVCache, ai: int, ao: torch.Tensor, ap: float | None = None) -> torch.Tensor:


    if ao.dim() != 2:
        raise ValueError(f"query must be 2-D [num_q_heads, head_dim], got {ao.dim()}-D")
    an, af = ao.shape


    if ap is not None:
        if isinstance(ap, torch.Tensor):
            ap = ap.item()
        if math.isnan(ap) or math.isinf(ap) or ap <= 0:
            raise ValueError(f"scale must be a positive finite number, got {ap}")
    ag = ap if ap is not None else 1.0 / (af ** 0.5)

    ah = aa.k_layers[ai]
    az = aa.v_layers[ai]
    ae = ao.device


    am = ah.num_heads

    if am <= 0:
        raise ValueError("num_kv_heads must be positive")

    if an % am != 0:
        raise ValueError(
            f"num_q_heads={an} must be divisible by "
            f"num_kv_heads={am}"
        )

    al = an // am


    if not ah.block_table and not ah._staging:
        raise ValueError(
            "paged_attention requires at least one cached token (both committed "
            "pages and staging are empty for this layer) -- did you forget to call "
            "cache.append(...) before paged_attention(...)? See this function's docstring."
        )

    if ah.block_table:
        z = torch.tensor(ah.block_table, dtype=torch.int32, ae=ae)


        aq = torch.tensor(
            [s if s is not None else 0.0 for s in ah.page_scales], dtype=torch.float32, ae=ae
        )
        ar = torch.tensor(
            [s if s is not None else 0.0 for s in az.page_scales], dtype=torch.float32, ae=ae
        )


        ad, ac, ab = _get_C().paged_attention_committed(
            ao, ah.storage, az.storage, aq, ar, z, ag
        )
    else:
        ab = torch.full((an,), float("-inf"), ae=ae)
        ac = torch.zeros(an, ae=ae)
        ad = torch.zeros(an, af, ae=ae)

    au = torch.stack(ah._staging) if ah._staging else torch.empty(0, am, af, ae=ae)
    ax = torch.stack(az._staging) if az._staging else torch.empty(0, am, af, ae=ae)


    au = _repeat_kv_heads(au, al)
    ax = _repeat_kv_heads(ax, al)
    av, aw, ay = _staging_contribution(ao, au, ax, ag)

    ak, aj = _merge_unnormalized(
        ab, ac, ad, av, aw, ay
    )
    return ak / aj.unsqueeze(-1)
