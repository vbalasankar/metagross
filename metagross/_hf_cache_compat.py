from __future__ import annotations

from collections.abc import Iterable

import torch
from transformers import DynamicCache


def extract_raw_kv(cache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:


    return tuple((b.keys, b.values) for b in cache.layers)


def build_dynamic_cache(
    raw_kv,
) -> DynamicCache:


    return DynamicCache(ddp_cache_data=tuple(raw_kv))
