from __future__ import annotations

import math

import torch

from .cache import PagedKVCache


def _get_C():
    from . import _C
    return _C


def _merge_unnormalized(
    max1, sum1, wv1: torch.Tensor,
    max2, sum2, wv2: torch.Tensor,
):


    m = torch.maximum(max1, max2)
    b = torch.exp(max1 - m)
    c = torch.exp(max2 - m)
    g = sum1 * b + sum2 * c
    h = wv1 * b.unsqueeze(-1) + wv2 * c.unsqueeze(-1)
    return h, g


def _staging_contribution(query, staging_k, staging_v, inv_sqrt_head_dim):


    o, l = query.shape
    k = query.device

    if staging_k.shape[0] == 0:


        return (
            torch.full((o,), float("-inf"), device=k),
            torch.zeros(o, device=k),
            torch.zeros(o, l, device=k),
        )


    q = torch.einsum("thd,hd->ht", staging_k, query) * inv_sqrt_head_dim
    m = q.max(dim=-1).values
    u = torch.exp(q - m.unsqueeze(-1))
    s = u.sum(dim=-1)
    wv = torch.einsum("ht,thd->hd", u, staging_v)
    return m, s, wv


def _repeat_kv_heads(x: torch.Tensor, n_rep) -> torch.Tensor:


    if n_rep == 1:
        return x
    seq, y, v = x.shape
    return x[:, :, None, :].expand(seq, y, n_rep, v).reshape(seq, y * n_rep, v)


def paged_attention(cache, layer_idx, query, scale = None) -> torch.Tensor:


    if query.dim() != 2:
        raise ValueError(f"query must be 2-D [num_q_heads, head_dim], got {query.dim()}-D")
    an, af = query.shape


    if scale is not None:
        if isinstance(scale, torch.Tensor):
            scale = scale.item()
        if math.isnan(scale) or math.isinf(scale) or scale <= 0:
            raise ValueError(f"scale must be a positive finite number, got {scale}")
    ag = scale if scale is not None else 1.0 / (af ** 0.5)

    ah = cache.k_layers[layer_idx]
    az = cache.v_layers[layer_idx]
    ae = query.device


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
        z = torch.tensor(ah.block_table, dtype=torch.int32, device=ae)


        aq = torch.tensor(
            [s if s is not None else 0.0 for s in ah.page_scales], dtype=torch.float32, device=ae
        )
        ar = torch.tensor(
            [s if s is not None else 0.0 for s in az.page_scales], dtype=torch.float32, device=ae
        )


        ad, ac, ab = _get_C().paged_attention_committed(
            query, ah.storage, az.storage, aq, ar, z, ag
        )
    else:
        ab = torch.full((an,), float("-inf"), device=ae)
        ac = torch.zeros(an, device=ae)
        ad = torch.zeros(an, af, device=ae)

    au = torch.stack(ah._staging) if ah._staging else torch.empty(0, am, af, device=ae)
    ax = torch.stack(az._staging) if az._staging else torch.empty(0, am, af, device=ae)


    au = _repeat_kv_heads(au, al)
    ax = _repeat_kv_heads(ax, al)
    av, aw, ay = _staging_contribution(query, au, ax, ag)

    ak, aj = _merge_unnormalized(
        ab, ac, ad, av, aw, ay
    )
    return ak / aj.unsqueeze(-1)
