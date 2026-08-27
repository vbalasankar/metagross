from __future__ import annotations

from collections.abc import Iterable

import torch
from transformers import DynamicCache


def extract_raw_kv(a) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:


    return tuple((b.keys, b.values) for b in a.layers)


def build_dynamic_cache(
    c: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> DynamicCache:


    return DynamicCache(ddp_cache_data=tuple(c))
