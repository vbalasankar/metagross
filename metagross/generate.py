from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hf_cache_compat import build_dynamic_cache, extract_raw_kv
from .attention import paged_attention
from .cache import PagedKVCache


def load_gpt2(device = "cuda"):
    c = AutoTokenizer.from_pretrained("gpt2")
    b = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16).to(device)
    b.eval()
    return b, c


@torch.no_grad()
def generate_metagross(
    model,
    tokenizer,
    prompt,
    max_new_tokens = 20,
    page_size = 16,
    max_pages = None,
):


    g = next(model.parameters()).device
    f = model.config
    aa = f.n_layer
    z = f.n_head
    j = f.n_embd // f.n_head

    l = tokenizer(prompt, return_tensors="pt").input_ids.to(g)
    af = l.shape[1]

    if max_pages is None:


        am = af + max_new_tokens
        max_pages = (am + page_size - 1) // page_size + 2

    u = PagedKVCache(
        num_layers=aa, num_heads=z, head_dim=j,
        page_size=page_size, max_pages=max_pages, device=g,
    )


    ab = model(input_ids=l, use_cache=True)
    d = [ab.logits[:, -1, :].clone()]

    ag = extract_raw_kv(ab.past_key_values)
    for r, (k, v) in enumerate(ag):


        m = k.squeeze(0).transpose(0, 1).contiguous().float()
        an = v.squeeze(0).transpose(0, 1).contiguous().float()
        u.append(r, m, an)
    del ab, ag

    y = torch.argmax(d[0], dim=-1, keepdim=True)
    i = [y.item()]


    for _ in range(max_new_tokens - 1):
        ai = u.seq_len


        ag = []
        for r in range(aa):
            n, ao = u.read_layer_fp32(r)
            o = n.transpose(0, 1).unsqueeze(0).to(torch.float16)
            ap = ao.transpose(0, 1).unsqueeze(0).to(torch.float16)
            ag.append((o, ap))
        ah = build_dynamic_cache(ag)

        ad = torch.tensor([[ai]], device=g)
        e = torch.ones((1, ai + 1), device=g, dtype=torch.long)

        ab = model(
            input_ids=y,
            past_key_values=ah,
            position_ids=ad,
            attention_mask=e,
            use_cache=True,
        )
        aj = ab.logits[:, -1, :]
        d.append(aj.clone())


        x = extract_raw_kv(ab.past_key_values)
        for r, (k, v) in enumerate(x):
            p = k[:, :, -1:, :].squeeze(0).transpose(0, 1).contiguous().float()
            aq = v[:, :, -1:, :].squeeze(0).transpose(0, 1).contiguous().float()
            u.append(r, p, aq)
        del ab, ah, x, ag

        y = torch.argmax(aj, dim=-1, keepdim=True)
        i.append(y.item())

    ak = tokenizer.decode(i)
    return ak, i, d, u


@torch.no_grad()
def generate_metagross_fused(
    model,
    tokenizer,
    prompt,
    max_new_tokens = 20,
    page_size = 16,
    max_pages = None,
):


    ax = next(model.parameters()).device
    aw = model.config
    bk = aw.n_layer
    bj = aw.n_head
    az = aw.n_embd // aw.n_head

    bb = tokenizer(prompt, return_tensors="pt").input_ids.to(ax)
    bp = bb.shape[1]

    if max_pages is None:
        bw = bp + max_new_tokens
        max_pages = (bw + page_size - 1) // page_size + 2

    av = PagedKVCache(
        num_layers=bk, num_heads=bj, head_dim=az,
        page_size=page_size, max_pages=max_pages, device=ax,
    )


    bl = model(input_ids=bb, use_cache=True)
    ar = [bl.logits[:, -1, :].clone()]

    bq = extract_raw_kv(bl.past_key_values)
    for bd, (k, v) in enumerate(bq):
        bc = k.squeeze(0).transpose(0, 1).contiguous().float()
        bx = v.squeeze(0).transpose(0, 1).contiguous().float()
        av.append(bd, bc, bx)
    del bl, bq

    bh = torch.argmax(ar[0], dim=-1, keepdim=True)
    ay = [bh.item()]


    for _ in range(max_new_tokens - 1):
        bs = av.seq_len
        bn = torch.tensor([[bs]], dtype=torch.long, device=ax)
        ba = model.transformer.wte(bh) + model.transformer.wpe(bn)

        for bd, au in enumerate(model.transformer.h):
            br = ba
            bi = au.ln_1(ba)

            qkv = au.attn.c_attn(bi)
            q, k, v = qkv.split(au.attn.split_size, dim=2)


            q = q.reshape(1, 1, bj, az).squeeze(0).squeeze(0).float()
            k = k.reshape(1, 1, bj, az).squeeze(0).float()
            v = v.reshape(1, 1, bj, az).squeeze(0).float()

            av.append(bd, k, v)
            at = paged_attention(av, bd, q, scale=au.attn.scaling)
            at = at.to(ba.dtype).reshape(1, 1, bj * az)
            at = au.attn.c_proj(at)

            ba = br + at

            br = ba
            bi = au.ln_2(ba)
            ba = br + au.mlp(bi)

        ba = model.transformer.ln_f(ba)
        bt = model.lm_head(ba[:, -1, :])
        ar.append(bt)

        bh = torch.argmax(bt, dim=-1, keepdim=True)
        ay.append(bh.item())

    bu = tokenizer.decode(ay)
    return bu, ay, ar, av


@torch.no_grad()
def generate_baseline_hf(model, tokenizer, prompt, max_new_tokens = 20):

    by = next(model.parameters()).device
    cc = tokenizer(prompt, return_tensors="pt").input_ids.to(by)

    cf = model(input_ids=cc, use_cache=True)
    bz = cf.logits[:, -1, :].clone()


    ca = model.generate(
        cc, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True,
    )
    cb = ca.sequences[0, cc.shape[1]:].tolist()
    ch = tokenizer.decode(cb)
    return ch, cb, bz
