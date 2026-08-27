from __future__ import annotations

import torch

from .allocator import BlockAllocator


def _get_C():


    from . import _C
    return _C


class LayerKVCache:


    def __init__(self, num_heads, head_dim, page_size, max_pages, device):


        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")
        if max_pages <= 0:
            raise ValueError(f"max_pages must be positive, got {max_pages}")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.device = device

        self.storage = torch.zeros(
            max_pages, page_size, num_heads, head_dim, dtype=torch.int8, device=device
        )
        self.page_scales: list[float | None] = [None] * max_pages
        self.allocator = BlockAllocator(max_pages)
        self.block_table: list[int] = []
        self.seq_len = 0
        self._staging: list[torch.Tensor] = []

    def append(self, new_values) -> None:


        if new_values.dim() != 3:
            raise ValueError(
                f"new_values must be 3-D [num_new_tokens, num_heads, head_dim], got {new_values.dim()}-D"
            )
        if new_values.shape[1] != self.num_heads:
            raise ValueError(f"expected num_heads={self.num_heads}, got {new_values.shape[1]}")
        if new_values.shape[2] != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}, got {new_values.shape[2]}")

        for i in range(new_values.shape[0]):
            self._staging.append(new_values[i])
            self.seq_len += 1
            if len(self._staging) == self.page_size:
                self._commit_staged_page()

    def _commit_staged_page(self) -> None:
        m = self.allocator.allocate()
        self.block_table.append(m)

        k = torch.stack(self._staging)
        l = self.storage[m]
        n = _get_C().quantize_new_scale(k, l, 0)
        self.page_scales[m] = n

        self._staging = []

    def read_fp32(self) -> torch.Tensor:

        p = []
        for q in self.block_table:
            r = self.page_scales[q]
            p.append(_get_C().dequantize_page(self.storage[q], 0, self.page_size, r))
        if self._staging:
            p.append(torch.stack(self._staging))

        if not p:
            return torch.empty(0, self.num_heads, self.head_dim, dtype=torch.float32, device=self.device)
        return torch.cat(p, dim=0)

    def memory_bytes(self) -> int:

        return self.storage.numel()


class PagedKVCache:


    def __init__(
        self, num_layers, num_heads, head_dim,
        page_size = 16, max_pages = 64, device = "cuda",
    ):


        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.num_layers = num_layers
        self.k_layers = [
            LayerKVCache(num_heads, head_dim, page_size, max_pages, device) for _ in range(num_layers)
        ]
        self.v_layers = [
            LayerKVCache(num_heads, head_dim, page_size, max_pages, device) for _ in range(num_layers)
        ]

    def append(self, layer_idx, new_k, new_v) -> None:
        if not (0 <= layer_idx < self.num_layers):
            raise ValueError(f"layer_idx={layer_idx} out of range for num_layers={self.num_layers}")


        if new_k.dim() != 3:
            raise ValueError(f"new_k must be 3-D [num_new_tokens, num_heads, head_dim], got {new_k.dim()}-D")
        if new_v.dim() != 3:
            raise ValueError(f"new_v must be 3-D [num_new_tokens, num_heads, head_dim], got {new_v.dim()}-D")

        if new_k.shape[0] != new_v.shape[0]:
            raise ValueError(
                f"K and V must have the same number of tokens, got {new_k.shape[0]} and {new_v.shape[0]}"
            )

        if new_k.shape[1:] != new_v.shape[1:]:
            raise ValueError(f"K and V shapes must match, got {tuple(new_k.shape)} and {tuple(new_v.shape)}")

        ac = self.k_layers[layer_idx].num_heads
        ab = self.k_layers[layer_idx].head_dim
        if new_k.shape[1] != ac:
            raise ValueError(f"expected num_heads={ac}, got {new_k.shape[1]}")
        if new_k.shape[2] != ab:
            raise ValueError(f"expected head_dim={ab}, got {new_k.shape[2]}")

        self.k_layers[layer_idx].append(new_k)
        self.v_layers[layer_idx].append(new_v)

    def read_layer_fp32(self, layer_idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_layers[layer_idx].read_fp32(), self.v_layers[layer_idx].read_fp32()

    @property
    def seq_len(self) -> int:
        return self.k_layers[0].seq_len

    def memory_bytes(self) -> int:

        return sum(ak.memory_bytes() for ak in self.k_layers + self.v_layers)
