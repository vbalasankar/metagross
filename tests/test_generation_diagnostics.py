import pytest
import torch

pytest.importorskip("metagross._C", reason="metagross._C extension not built yet")

from metagross._hf_cache_compat import extract_raw_kv
from metagross.attention import paged_attention
from metagross.cache import PagedKVCache
from metagross.generate import generate_metagross, generate_metagross_fused, load_gpt2


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")

PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "def fibonacci(n):\n    if n <= 1:",
]

TEST_PAGE_SIZE = 4


BPRIME_VS_CPRIME_MAX_ATTN_DIFF = 0.5
BPRIME_VS_CPRIME_MAX_LOGIT_DIFF = 1.0


@pytest.fixture(scope="module")
def gpt2():
    return load_gpt2(device="cuda")


@torch.no_grad()
def _true_step2_reference(model, tokenizer, prompt, device):


    o = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    p = model(input_ids=o, use_cache=True)
    l = torch.argmax(p.logits[:, -1, :], dim=-1, keepdim=True)
    i = torch.cat([o, l], dim=1)

    d = model(input_ids=i, use_cache=True)
    x = model(input_ids=i, use_cache=False)
    c = (d.logits[:, -1, :] - x.logits[:, -1, :]).abs().max().item()

    u = x.logits[:, -1, :]
    w = extract_raw_kv(d.past_key_values)
    return u, w, c


def _kv_error_vs_true(metagross_cache, true_raw_kv, num_layers):


    ah = []
    ab, am = [], []
    for ae in range(num_layers):
        ac, an = metagross_cache.read_layer_fp32(ae)
        ad = ac.transpose(0, 1).unsqueeze(0).float()
        ao = an.transpose(0, 1).unsqueeze(0).float()
        ai, ak = true_raw_kv[ae]
        aa = (ad - ai.float()).abs()
        al = (ao - ak.float()).abs()
        ah.append((ae, aa.max().item(), aa.mean().item(), al.max().item(), al.mean().item()))
        ab.append(aa)
        am.append(al)
    y = torch.cat([e.flatten() for e in ab])
    z = torch.cat([e.flatten() for e in am])
    return ah, y.max().item(), y.mean().item(), z.max().item(), z.mean().item()


def _print_kv_error_table(label, rows, agg):
    ap, aq, ay, az = agg
    print(f"\n  {label} K/V error vs TRUE (unquantized) K/V, per layer:")
    print(f"  {'layer':>5}  {'K max':>10}  {'K mean':>10}  {'V max':>10}  {'V mean':>10}")
    for aw, ar, au, ba, bb in rows:
        print(f"  {aw:>5}  {ar:>10.5f}  {au:>10.5f}  {ba:>10.5f}  {bb:>10.5f}")
    print(f"  AGGREGATE (all {len(rows)} layers): K max={ap:.5f} mean={aq:.5f} | V max={ay:.5f} mean={az:.5f}")


def _print_attn_error_table(label, per_layer_attn_a, per_layer_attn_b):
    print(f"\n  {label} per-layer attention-OUTPUT error:")
    print(f"  {'layer':>5}  {'max':>10}  {'mean':>10}")
    be, bf = [], []
    for bd, (a, b) in enumerate(zip(per_layer_attn_a, per_layer_attn_b)):
        err = (a.float() - b.float()).abs()
        be.append(err.max().item())
        bf.append(err.mean().item())
        print(f"  {bd:>5}  {be[-1]:>10.5f}  {bf[-1]:>10.5f}")
    print(f"  AGGREGATE (all {len(be)} layers): max={max(be):.5f} mean={sum(bf) / len(bf):.5f}")
    return max(be), sum(bf) / len(bf)


@torch.no_grad()
def _generate_gpt2_reference_attention(model, tokenizer, prompt, max_new_tokens, page_size, max_pages=None):


    bo = next(model.parameters()).device
    bn = model.config
    cd = bn.n_layer
    cc = bn.n_head
    bq = bn.n_embd // bn.n_head

    bs = tokenizer(prompt, return_tensors="pt").input_ids.to(bo)
    ci = bs.shape[1]
    if max_pages is None:
        max_pages = (ci + max_new_tokens + page_size - 1) // page_size + 2

    bm = PagedKVCache(num_layers=cd, num_heads=cc, head_dim=bq,
                          page_size=page_size, max_pages=max_pages, device=bo)

    ce = model(input_ids=bs, use_cache=True)
    bi = [ce.logits[:, -1, :].clone()]
    cj = extract_raw_kv(ce.past_key_values)
    for bv, (k, v) in enumerate(cj):
        bt = k.squeeze(0).transpose(0, 1).contiguous().float()
        cr = v.squeeze(0).transpose(0, 1).contiguous().float()
        bm.append(bv, bt, cr)
    del ce, cj

    ca = torch.argmax(bi[0], dim=-1, keepdim=True)
    bp = [ca.item()]
    bk = []

    for _ in range(max_new_tokens - 1):
        cn = bm.seq_len
        cg = torch.tensor([[cn]], dtype=torch.long, device=bo)
        br = model.transformer.wte(ca) + model.transformer.wpe(cg)

        bk = []
        for bv, bl in enumerate(model.transformer.h):
            ck = br
            cb = bl.ln_1(br)

            qkv = bl.attn.c_attn(cb)
            q, k, v = qkv.split(bl.attn.split_size, dim=2)
            q = q.reshape(1, 1, cc, bq).squeeze(0).squeeze(0).float()
            k = k.reshape(1, 1, cc, bq).squeeze(0).float()
            v = v.reshape(1, 1, cc, bq).squeeze(0).float()

            bm.append(bv, k, v)


            bu, cs = bm.read_layer_fp32(bv)
            cl = bl.attn.scaling
            cm = torch.einsum("thd,hd->ht", bu, q) * cl
            cm = cm - cm.max(dim=-1, keepdim=True).values
            ct = torch.softmax(cm, dim=-1)
            bj = torch.einsum("ht,thd->hd", ct, cs)


            bk.append(bj.detach().clone())

            bj = bj.to(br.dtype).reshape(1, 1, cc * bq)
            bj = bl.attn.c_proj(bj)

            br = ck + bj

            ck = br
            cb = bl.ln_2(br)
            br = ck + bl.mlp(cb)

        br = model.transformer.ln_f(br)
        co = model.lm_head(br[:, -1, :])
        bi.append(co)

        ca = torch.argmax(co, dim=-1, keepdim=True)
        bp.append(ca.item())

    cp = tokenizer.decode(bp)
    return cp, bp, bi, bm, bk


@torch.no_grad()
def _generate_gpt2_fused_instrumented(model, tokenizer, prompt, max_new_tokens, page_size, max_pages=None):


    db = next(model.parameters()).device
    da = model.config
    do = da.n_layer
    dn = da.n_head
    dd = da.n_embd // da.n_head

    df = tokenizer(prompt, return_tensors="pt").input_ids.to(db)
    dt = df.shape[1]
    if max_pages is None:
        max_pages = (dt + max_new_tokens + page_size - 1) // page_size + 2

    cz = PagedKVCache(num_layers=do, num_heads=dn, head_dim=dd,
                          page_size=page_size, max_pages=max_pages, device=db)

    dp = model(input_ids=df, use_cache=True)
    cv = [dp.logits[:, -1, :].clone()]
    du = extract_raw_kv(dp.past_key_values)
    for dh, (k, v) in enumerate(du):
        dg = k.squeeze(0).transpose(0, 1).contiguous().float()
        ea = v.squeeze(0).transpose(0, 1).contiguous().float()
        cz.append(dh, dg, ea)
    del dp, du

    dl = torch.argmax(cv[0], dim=-1, keepdim=True)
    dc = [dl.item()]
    cx = []

    for _ in range(max_new_tokens - 1):
        dw = cz.seq_len
        dr = torch.tensor([[dw]], dtype=torch.long, device=db)
        de = model.transformer.wte(dl) + model.transformer.wpe(dr)

        cx = []
        for dh, cy in enumerate(model.transformer.h):
            dv = de
            dm = cy.ln_1(de)

            qkv = cy.attn.c_attn(dm)
            q, k, v = qkv.split(cy.attn.split_size, dim=2)
            q = q.reshape(1, 1, dn, dd).squeeze(0).squeeze(0).float()
            k = k.reshape(1, 1, dn, dd).squeeze(0).float()
            v = v.reshape(1, 1, dn, dd).squeeze(0).float()

            cz.append(dh, k, v)
            cw = paged_attention(cz, dh, q, scale=cy.attn.scaling)

            cx.append(cw.detach().clone())

            cw = cw.to(de.dtype).reshape(1, 1, dn * dd)
            cw = cy.attn.c_proj(cw)

            de = dv + cw

            dv = de
            dm = cy.ln_2(de)
            de = dv + cy.mlp(dm)

        de = model.transformer.ln_f(de)
        dx = model.lm_head(de[:, -1, :])
        cv.append(dx)

        dl = torch.argmax(dx, dim=-1, keepdim=True)
        dc.append(dl.item())

    dy = tokenizer.decode(dc)
    return dy, dc, cv, cz, cx


@pytest.mark.parametrize("prompt", PROMPTS)
def test_three_stage_diagnostic(gpt2, prompt):
    fi, fn = gpt2
    fa = next(fi.parameters()).device
    fj = fi.config.n_layer

    fo, fp, ez = _true_step2_reference(fi, fn, prompt, fa)

    _, _, fh, fg = generate_metagross(
        fi, fn, prompt, max_new_tokens=2, page_size=TEST_PAGE_SIZE
    )
    ei = fh[1]

    _, _, fe, fb = generate_metagross_fused(
        fi, fn, prompt, max_new_tokens=2, page_size=TEST_PAGE_SIZE
    )
    et = fe[1]

    _, _, fm, fl, en = _generate_gpt2_reference_attention(
        fi, fn, prompt, max_new_tokens=2, page_size=TEST_PAGE_SIZE
    )
    eo = fm[1]

    _, _, fd, fc, ev = _generate_gpt2_fused_instrumented(
        fi, fn, prompt, max_new_tokens=2, page_size=TEST_PAGE_SIZE
    )
    ew = fd[1]

    ej, *eh = _kv_error_vs_true(fg, fp, fj)
    eu, *es = _kv_error_vs_true(fb, fp, fj)
    ep, *em = _kv_error_vs_true(fl, fp, fj)

    eb = (fo - ei).abs().max().item()
    ec = (fo - ei).abs().mean().item()
    ef = (fo - et).abs().max().item()
    eg = (fo - et).abs().mean().item()
    ed = (fo - eo).abs().max().item()
    ee = (fo - eo).abs().mean().item()
    ek = (ei - et).abs().max().item()
    el = (ei - et).abs().mean().item()
    eq = (eo - ew).abs().max().item()
    er = (eo - ew).abs().mean().item()

    print(f"\n{'=' * 78}\n[{prompt!r}]\n{'=' * 78}")
    print(f"use_cache=True vs False consistency check (expect ~FP16 noise only): {ez:.5f}")
    _print_kv_error_table("Phase 1 (generate_metagross)", ej, eh)
    _print_kv_error_table("Phase 2 (generate_metagross_fused)", eu, es)
    _print_kv_error_table("B' (isolated reference-attention)", ep, em)

    ex, ey = _print_attn_error_table("B' vs C' (isolated ref-attn vs kernel)", en, ev)

    print(f"\nLogit differences:")
    print(f"  A vs B      (true vs Phase 1        -- pure quantization, whole-model):  max={eb:.5f} mean={ec:.5f}")
    print(f"  A vs C      (true vs Phase 2         -- quantization + kernel combined): max={ef:.5f} mean={eg:.5f}")
    print(f"  A vs B'     (true vs isolated ref-attn -- pure quantization, isolated):  max={ed:.5f} mean={ee:.5f}")
    print(f"  B vs C      (Phase 1 vs Phase 2, informational only):                    max={ek:.5f} mean={el:.5f}")
    print(f"  B' vs C'    (isolated ref-attn vs kernel -- KERNEL CORRECTNESS, gated):  max={eq:.5f} mean={er:.5f}")

    assert torch.isfinite(fo).all(), f"[{prompt!r}] NaN/Inf in true_logits -- baseline itself broken, unrelated to Metagross"
    assert torch.isfinite(ei).all(), f"[{prompt!r}] NaN/Inf in Phase 1 logits -- a real bug, not quantization noise"
    assert torch.isfinite(et).all(), f"[{prompt!r}] NaN/Inf in Phase 2 logits -- a real bug, not quantization noise"
    assert torch.isfinite(eo).all(), f"[{prompt!r}] NaN/Inf in isolated reference-attention logits"
    assert torch.isfinite(ew).all(), f"[{prompt!r}] NaN/Inf in instrumented-kernel logits"


    assert ex < BPRIME_VS_CPRIME_MAX_ATTN_DIFF, (
        f"[{prompt!r}] isolated reference attention (B') and the fused CUDA kernel (C') produce different "
        f"per-layer attention OUTPUTS (max diff {ex:.4f}) even though both read the IDENTICAL "
        f"dequantized cache contents at each layer -- see the per-layer table printed above for exactly "
        f"which layer first diverges; that layer's metagross.attention.paged_attention call (and, by "
        f"extension, csrc/paged_attention.cu) is where to look, not quantization."
    )
    assert eq < BPRIME_VS_CPRIME_MAX_LOGIT_DIFF, (
        f"[{prompt!r}] isolated reference attention (B') and the fused CUDA kernel (C') diverge by "
        f"{eq:.4f} in final logits even though both ran the IDENTICAL manual "
        f"per-layer loop over the IDENTICAL dequantized cache contents -- see the attention-output table "
        f"above to find which layer's output first diverges before this compounded into the final logits."
    )
