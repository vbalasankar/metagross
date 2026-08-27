from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hf_cache_compat import extract_raw_kv
from .attention import paged_attention
from .cache import PagedKVCache

TINYLLAMA_CHECKPOINT = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_tinyllama(a: str = "cuda"):
    c = AutoTokenizer.from_pretrained(TINYLLAMA_CHECKPOINT)
    b = AutoModelForCausalLM.from_pretrained(TINYLLAMA_CHECKPOINT, torch_dtype=torch.float16).to(a)
    b.eval()
    return b, c


def _rope_cos_sin(i: int, g: int, j: float, d, e):


    h = 1.0 / (j ** (torch.arange(0, g, 2, e=torch.float32, d=d) / g))
    f = i * h
    emb = torch.cat([f, f])
    return emb.cos().to(e), emb.sin().to(e)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    l = x.shape[-1] // 2
    x1, x2 = x[..., :l], x[..., l:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:

    return x * cos + _rotate_half(x) * sin


@torch.no_grad()
def generate_tinyllama_fused(
    ae,
    av,
    am: str,
    ac: int = 20,
    al: int = 16,
    ad: int | None = None,
):


    r = next(ae.parameters()).device
    p = ae.config
    ai = p.num_hidden_layers
    aj = p.num_attention_heads
    ah = p.num_key_value_heads
    t = getattr(p, "head_dim", p.hidden_size // aj)
    aq = getattr(p, "rope_theta", 10000.0)

    w = av(am, return_tensors="pt").input_ids.to(r)
    an = w.shape[1]

    if ad is None:
        aw = an + ac
        ad = (aw + al - 1) // al + 2


    o = PagedKVCache(
        ai=ai, num_heads=ah, t=t,
        al=al, ad=ad, r=r,
    )


    ak = ae(w=w, use_cache=True)
    m = [ak.logits[:, -1, :].clone()]

    ao = extract_raw_kv(ak.past_key_values)
    for ab, (k, v) in enumerate(ao):


        y = k.squeeze(0).transpose(0, 1).contiguous().float()
        ax = v.squeeze(0).transpose(0, 1).contiguous().float()
        o.append(ab, y, ax)
    del ak, ao

    af = torch.argmax(m[0], dim=-1, keepdim=True)
    s = [af.item()]


    for _ in range(ac - 1):
        ar = o.seq_len
        u = ae.model.embed_tokens(af)

        cos, sin = _rope_cos_sin(ar, t, aq, r, u.dtype)

        for ab, aa in enumerate(ae.model.layers):
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

        u = ae.model.norm(u)
        at = ae.lm_head(u[:, -1, :])
        m.append(at)

        af = torch.argmax(at, dim=-1, keepdim=True)
        s.append(af.item())

    au = av.decode(s)
    return au, s, m, o
