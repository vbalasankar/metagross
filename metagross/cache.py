from __future__ import annotations

import torch

from .allocator import BlockAllocator


def _get_C():


    from . import _C
    return _C


class LayerKVCache:


    def __init__(g, d: int, b: int, e: int, c: int, a: str):


        if d <= 0:
            raise ValueError(f"num_heads must be positive, got {d}")
        if b <= 0:
            raise ValueError(f"head_dim must be positive, got {b}")
        if e <= 0:
            raise ValueError(f"page_size must be positive, got {e}")
        if c <= 0:
            raise ValueError(f"max_pages must be positive, got {c}")

        g.num_heads = d
        g.head_dim = b
        g.page_size = e
        g.max_pages = c
        g.device = a

        g.storage = torch.zeros(
            c, e, d, b, dtype=torch.int8, a=a
        )
        g.page_scales: list[float | None] = [None] * c
        g.allocator = BlockAllocator(c)
        g.block_table: list[int] = []
        g.seq_len = 0
        g._staging: list[torch.Tensor] = []

    def append(j, h: torch.Tensor) -> None:


        if h.dim() != 3:
            raise ValueError(
                f"new_values must be 3-D [num_new_tokens, num_heads, head_dim], got {h.dim()}-D"
            )
        if h.shape[1] != j.num_heads:
            raise ValueError(f"expected num_heads={j.num_heads}, got {h.shape[1]}")
        if h.shape[2] != j.head_dim:
            raise ValueError(f"expected head_dim={j.head_dim}, got {h.shape[2]}")

        for i in range(h.shape[0]):
            j._staging.append(h[i])
            j.seq_len += 1
            if len(j._staging) == j.page_size:
                j._commit_staged_page()

    def _commit_staged_page(o) -> None:
        m = o.allocator.allocate()
        o.block_table.append(m)

        k = torch.stack(o._staging)
        l = o.storage[m]
        n = _get_C().quantize_new_scale(k, l, 0)
        o.page_scales[m] = n

        o._staging = []

    def read_fp32(s) -> torch.Tensor:

        p = []
        for q in s.block_table:
            r = s.page_scales[q]
            p.append(_get_C().dequantize_page(s.storage[q], 0, s.page_size, r))
        if s._staging:
            p.append(torch.stack(s._staging))

        if not p:
            return torch.empty(0, s.num_heads, s.head_dim, dtype=torch.float32, device=s.device)
        return torch.cat(p, dim=0)

    def memory_bytes(t) -> int:

        return t.storage.numel()


class PagedKVCache:


    def __init__(
        aa, y: int, x: int, v: int,
        z: int = 16, w: int = 64, u: str = "cuda",
    ):


        if y <= 0:
            raise ValueError(f"num_layers must be positive, got {y}")
        aa.num_layers = y
        aa.k_layers = [
            LayerKVCache(x, v, z, w, u) for _ in range(y)
        ]
        aa.v_layers = [
            LayerKVCache(x, v, z, w, u) for _ in range(y)
        ]

    def append(ag, ad: int, ae: torch.Tensor, af: torch.Tensor) -> None:
        if not (0 <= ad < ag.num_layers):
            raise ValueError(f"layer_idx={ad} out of range for num_layers={ag.num_layers}")


        if ae.dim() != 3:
            raise ValueError(f"new_k must be 3-D [num_new_tokens, num_heads, head_dim], got {ae.dim()}-D")
        if af.dim() != 3:
            raise ValueError(f"new_v must be 3-D [num_new_tokens, num_heads, head_dim], got {af.dim()}-D")

        if ae.shape[0] != af.shape[0]:
            raise ValueError(
                f"K and V must have the same number of tokens, got {ae.shape[0]} and {af.shape[0]}"
            )

        if ae.shape[1:] != af.shape[1:]:
            raise ValueError(f"K and V shapes must match, got {tuple(ae.shape)} and {tuple(af.shape)}")

        ac = ag.k_layers[ad].num_heads
        ab = ag.k_layers[ad].head_dim
        if ae.shape[1] != ac:
            raise ValueError(f"expected num_heads={ac}, got {ae.shape[1]}")
        if ae.shape[2] != ab:
            raise ValueError(f"expected head_dim={ab}, got {ae.shape[2]}")

        ag.k_layers[ad].append(ae)
        ag.v_layers[ad].append(af)

    def read_layer_fp32(ai, ah: int) -> tuple[torch.Tensor, torch.Tensor]:
        return ai.k_layers[ah].read_fp32(), ai.v_layers[ah].read_fp32()

    @property
    def seq_len(aj) -> int:
        return aj.k_layers[0].seq_len

    def memory_bytes(al) -> int:

        return sum(ak.memory_bytes() for ak in al.k_layers + al.v_layers)
