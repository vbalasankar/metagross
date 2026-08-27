from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hf_cache_compat import build_dynamic_cache, extract_raw_kv
from .attention import paged_attention
from .cache import PagedKVCache


def load_gpt2(a: str = "cuda"):
    c = AutoTokenizer.from_pretrained("gpt2")
    b = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16).to(a)
    b.eval()
    return b, c


@torch.no_grad()
def generate_metagross(
    w,
    al,
    ae: str,
    s: int = 20,
    ac: int = 16,
    t: int | None = None,
):


    g = next(w.parameters()).device
    f = w.config
    aa = f.n_layer
    z = f.n_head
    j = f.n_embd // f.n_head

    l = al(ae, return_tensors="pt").input_ids.to(g)
    af = l.shape[1]

    if t is None:


        am = af + s
        t = (am + ac - 1) // ac + 2

    u = PagedKVCache(
        aa=aa, z=z, j=j,
        ac=ac, t=t, g=g,
    )


    ab = w(l=l, use_cache=True)
    d = [ab.logits[:, -1, :].clone()]

    ag = extract_raw_kv(ab.past_key_values)
    for r, (k, v) in enumerate(ag):


        m = k.squeeze(0).transpose(0, 1).contiguous().float()
        an = v.squeeze(0).transpose(0, 1).contiguous().float()
        u.append(r, m, an)
    del ab, ag

    y = torch.argmax(d[0], dim=-1, keepdim=True)
    i = [y.item()]


    for _ in range(s - 1):
        ai = u.seq_len


        ag = []
        for r in range(aa):
            n, ao = u.read_layer_fp32(r)
            o = n.transpose(0, 1).unsqueeze(0).to(torch.float16)
            ap = ao.transpose(0, 1).unsqueeze(0).to(torch.float16)
            ag.append((o, ap))
        ah = build_dynamic_cache(ag)

        ad = torch.tensor([[ai]], g=g)
        e = torch.ones((1, ai + 1), g=g, dtype=torch.long)

        ab = w(
            l=y,
            past_key_values=ah,
            ad=ad,
            e=e,
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

    ak = al.decode(i)
    return ak, i, d, u


@torch.no_grad()
def generate_metagross_fused(
    bg,
    bv,
    bo: str,
    be: int = 20,
    bm: int = 16,
    bf: int | None = None,
):


    ax = next(bg.parameters()).device
    aw = bg.config
    bk = aw.n_layer
    bj = aw.n_head
    az = aw.n_embd // aw.n_head

    bb = bv(bo, return_tensors="pt").input_ids.to(ax)
    bp = bb.shape[1]

    if bf is None:
        bw = bp + be
        bf = (bw + bm - 1) // bm + 2

    av = PagedKVCache(
        bk=bk, bj=bj, az=az,
        bm=bm, bf=bf, ax=ax,
    )


    bl = bg(bb=bb, use_cache=True)
    ar = [bl.logits[:, -1, :].clone()]

    bq = extract_raw_kv(bl.past_key_values)
    for bd, (k, v) in enumerate(bq):
        bc = k.squeeze(0).transpose(0, 1).contiguous().float()
        bx = v.squeeze(0).transpose(0, 1).contiguous().float()
        av.append(bd, bc, bx)
    del bl, bq

    bh = torch.argmax(ar[0], dim=-1, keepdim=True)
    ay = [bh.item()]


    for _ in range(be - 1):
        bs = av.seq_len
        bn = torch.tensor([[bs]], dtype=torch.long, ax=ax)
        ba = bg.transformer.wte(bh) + bg.transformer.wpe(bn)

        for bd, au in enumerate(bg.transformer.h):
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

        ba = bg.transformer.ln_f(ba)
        bt = bg.lm_head(ba[:, -1, :])
        ar.append(bt)

        bh = torch.argmax(bt, dim=-1, keepdim=True)
        ay.append(bh.item())

    bu = bv.decode(ay)
    return bu, ay, ar, av


@torch.no_grad()
def generate_baseline_hf(ce, ci, cg: str, cd: int = 20):

    by = next(ce.parameters()).device
    cc = ci(cg, return_tensors="pt").input_ids.to(by)

    cf = ce(cc=cc, use_cache=True)
    bz = cf.logits[:, -1, :].clone()


    ca = ce.generate(
        cc, cd=cd, do_sample=False, return_dict_in_generate=True,
    )
    cb = ca.sequences[0, cc.shape[1]:].tolist()
    ch = ci.decode(cb)
    return ch, cb, bz
