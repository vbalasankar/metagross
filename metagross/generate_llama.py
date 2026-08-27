from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hf_cache_compat import extract_raw_kv
from .attention import paged_attention
from .cache import PagedKVCache

TINYLLAMA_CHECKPOINT = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_tinyllama(device = "cuda"):
    c = AutoTokenizer.from_pretrained(TINYLLAMA_CHECKPOINT)
    b = AutoModelForCausalLM.from_pretrained(TINYLLAMA_CHECKPOINT, torch_dtype=torch.float16).to(device)
    b.eval()
    return b, c


def _rope_cos_sin(position, head_dim, rope_theta, device, dtype):


    h = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    f = position * h
    emb = torch.cat([f, f])
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    l = x.shape[-1] // 2
    x1, x2 = x[..., :l], x[..., l:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:

    return x * cos + _rotate_half(x) * sin


@torch.no_grad()
def generate_tinyllama_fused(
    model,
    tokenizer,
    prompt,
    max_new_tokens = 20,
    page_size = 16,
    max_pages = None,
):


    r = next(model.parameters()).device
    p = model.config
    ai = p.num_hidden_layers
    aj = p.num_attention_heads
    ah = p.num_key_value_heads
    t = getattr(p, "head_dim", p.hidden_size // aj)
    aq = getattr(p, "rope_theta", 10000.0)

    w = tokenizer(prompt, return_tensors="pt").input_ids.to(r)
    an = w.shape[1]

    if max_pages is None:
        aw = an + max_new_tokens
        max_pages = (aw + page_size - 1) // page_size + 2


    o = PagedKVCache(
        num_layers=ai, num_heads=ah, head_dim=t,
        page_size=page_size, max_pages=max_pages, device=r,
    )


    ak = model(input_ids=w, use_cache=True)
    m = [ak.logits[:, -1, :].clone()]

    ao = extract_raw_kv(ak.past_key_values)
    for ab, (k, v) in enumerate(ao):


        y = k.squeeze(0).transpose(0, 1).contiguous().float()
        ax = v.squeeze(0).transpose(0, 1).contiguous().float()
        o.append(ab, y, ax)
    del ak, ao

    af = torch.argmax(m[0], dim=-1, keepdim=True)
    s = [af.item()]


    for _ in range(max_new_tokens - 1):
        ar = o.seq_len
        u = model.model.embed_tokens(af)

        cos, sin = _rope_cos_sin(ar, t, aq, r, u.dtype)

        for ab, aa in enumerate(model.model.layers):
            ap = u
            ag = aa.input_layernorm(u)

            q = aa.self_attn.q_proj(ag).reshape(1, 1, aj, t).squeeze(0).squeeze(0).float()
            k = aa.self_attn.k_proj(ag).reshape(1, 1, ah, t).squeeze(0).float()
            v = aa.self_attn.v_proj(ag).reshape(1, 1, ah, t).squeeze(0).float()


            q = _apply_rope(q, cos, sin)
            z = _apply_rope(k.squeeze(0), cos, sin).unsqueeze(0)

            o.append(ab, z, v)
            n = paged_attention(o, ab, q, scale=aa.self_attn.scaling)
            n = n.to(u.dtype).reshape(1, 1, aj * t)
            n = aa.self_attn.o_proj(n)

            u = ap + n

            ap = u
            ag = aa.post_attention_layernorm(u)
            u = ap + aa.mlp(ag)

        u = model.model.norm(u)
        at = model.lm_head(u[:, -1, :])
        m.append(at)

        af = torch.argmax(at, dim=-1, keepdim=True)
        s.append(af.item())

    au = tokenizer.decode(s)
    return au, s, m, o
