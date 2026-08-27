import pytest
import torch

pytest.importorskip("metagross._C", reason="metagross._C extension not built yet")

from metagross import _C
from metagross.attention import _merge_unnormalized, _repeat_kv_heads, _staging_contribution, paged_attention
from metagross.cache import PagedKVCache


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")


def _reference_full_attention(query, k, v, inv_sqrt_head_dim):


    d = torch.einsum("thd,hd->ht", k, query) * inv_sqrt_head_dim
    d = d - d.max(dim=-1, keepdim=True).values
    e = torch.softmax(d, dim=-1)
    return torch.einsum("ht,thd->hd", e, v)


def _make_cache_with_tokens(num_layers, num_heads, head_dim, page_size, max_pages, k_tokens, v_tokens, device="cuda"):
    f = PagedKVCache(num_layers=num_layers, num_heads=num_heads, head_dim=head_dim,
                          page_size=page_size, max_pages=max_pages, device=device)
    for j in range(num_layers):
        f.append(j, k_tokens, v_tokens)
    return f


class TestPagedAttentionKernelDirectly:


    def test_matches_reference_when_everything_is_committed(self):
        torch.manual_seed(0)
        ad, u, ae = 4, 64, 16
        ab = 3
        ac = ab * ae

        t = _make_cache_with_tokens(
            1, ad, u, ae, max_pages=ab,
            k_tokens=torch.randn(ac, ad, u, device="cuda"),
            v_tokens=torch.randn(ac, ad, u, device="cuda"),
        )
        y, am = t.k_layers[0], t.v_layers[0]
        assert not y._staging

        af = torch.randn(ad, u, device="cuda")
        w = 1.0 / (u ** 0.5)

        r = torch.tensor(y.block_table, dtype=torch.int32, device="cuda")


        aj = torch.tensor(
            [s if s is not None else 0.0 for s in y.page_scales], device="cuda"
        )
        ak = torch.tensor(
            [s if s is not None else 0.0 for s in am.page_scales], device="cuda"
        )

        ao, an, aa = _C.paged_attention_committed(
            af, y.storage, am.storage, aj, ak, r, w
        )
        z = ao / an.unsqueeze(-1)


        ag = y.read_fp32()
        ah = am.read_fp32()
        ai = _reference_full_attention(af, ag, ah, w)

        torch.testing.assert_close(z, ai, atol=1e-3, rtol=1e-3)


class TestStagingContribution:


    def test_matches_reference(self):
        torch.manual_seed(1)
        at, ap, ar = 4, 64, 7
        au = torch.randn(at, ap, device="cuda")
        k = torch.randn(ar, at, ap, device="cuda")
        v = torch.randn(ar, at, ap, device="cuda")
        aq = 1.0 / (ap ** 0.5)

        m, s, wv = _staging_contribution(au, k, v, aq)
        got = wv / s.unsqueeze(-1)
        av = _reference_full_attention(au, k, v, aq)
        torch.testing.assert_close(got, av, atol=1e-4, rtol=1e-4)

    def test_empty_staging_returns_identity(self):
        az = torch.randn(4, 64, device="cuda")
        ax = torch.empty(0, 4, 64, device="cuda")
        ay = torch.empty(0, 4, 64, device="cuda")
        m, s, wv = _staging_contribution(az, ax, ay, 1.0 / 8.0)
        assert torch.isinf(m).all() and (m < 0).all()
        assert (s == 0).all()
        assert (wv == 0).all()


class TestMergeUnnormalized:


    def test_merging_with_identity_is_a_noop(self):
        torch.manual_seed(2)
        bi, bb = 4, 64
        bf = torch.randn(bi, device="cuda")
        bk = torch.rand(bi, device="cuda") + 0.1
        wv2 = torch.randn(bi, bb, device="cuda")

        bc = torch.full((bi,), float("-inf"), device="cuda")
        bd = torch.zeros(bi, device="cuda")
        be = torch.zeros(bi, bb, device="cuda")

        bh, bg = _merge_unnormalized(bc, bd, be, bf, bk, wv2)
        torch.testing.assert_close(bg, bk)
        torch.testing.assert_close(bh, wv2)

    def test_merge_matches_reference_full_attention(self):
        torch.manual_seed(3)
        bp, bl = 4, 64
        n1, n2 = 10, 6
        bq = torch.randn(bp, bl, device="cuda")
        k1 = torch.randn(n1, bp, bl, device="cuda")
        v1 = torch.randn(n1, bp, bl, device="cuda")
        k2 = torch.randn(n2, bp, bl, device="cuda")
        v2 = torch.randn(n2, bp, bl, device="cuda")
        bm = 1.0 / (bl ** 0.5)

        m1, s1, wv1 = _staging_contribution(bq, k1, v1, bm)
        m2, s2, wv2 = _staging_contribution(bq, k2, v2, bm)
        bo, bn = _merge_unnormalized(m1, s1, wv1, m2, s2, wv2)
        got = bo / bn.unsqueeze(-1)

        br = _reference_full_attention(bq, torch.cat([k1, k2]), torch.cat([v1, v2]), bm)
        torch.testing.assert_close(got, br, atol=1e-4, rtol=1e-4)


class TestPagedAttentionEndToEnd:


    def test_matches_reference_with_mixed_committed_and_staged_tokens(self):
        torch.manual_seed(4)
        cb, by, cc = 4, 64, 16
        bz = 16
        ca = 5

        bt = torch.randn(bz + ca, cb, by, device="cuda")
        bu = torch.randn(bz + ca, cb, by, device="cuda")

        bv = PagedKVCache(num_layers=1, num_heads=cb, head_dim=by,
                              page_size=cc, max_pages=4, device="cuda")
        bv.append(0, bt, bu)
        assert len(bv.k_layers[0].block_table) == 1
        assert len(bv.k_layers[0]._staging) == ca

        cd = torch.randn(cb, by, device="cuda")
        got = paged_attention(bv, layer_idx=0, query=cd)


        ce = bv.k_layers[0].read_fp32()[:bz]
        cf = bv.v_layers[0].read_fp32()[:bz]
        ci = torch.stack(bv.k_layers[0]._staging)
        cj = torch.stack(bv.v_layers[0]._staging)
        bw = torch.cat([ce, ci])
        bx = torch.cat([cf, cj])
        cg = _reference_full_attention(cd, bw, bx, 1.0 / (by ** 0.5))

        torch.testing.assert_close(got, cg, atol=1e-3, rtol=1e-3)

    def test_matches_reference_with_nothing_committed_yet(self):


        torch.manual_seed(5)
        co, cl, cp = 4, 64, 16
        cn = 5

        cm = torch.randn(cn, co, cl, device="cuda")
        ct = torch.randn(cn, co, cl, device="cuda")
        ck = PagedKVCache(num_layers=1, num_heads=co, head_dim=cl,
                              page_size=cp, max_pages=4, device="cuda")
        ck.append(0, cm, ct)
        assert ck.k_layers[0].block_table == []

        cq = torch.randn(co, cl, device="cuda")
        got = paged_attention(ck, layer_idx=0, query=cq)
        cr = _reference_full_attention(cq, cm, ct, 1.0 / (cl ** 0.5))
        torch.testing.assert_close(got, cr, atol=1e-4, rtol=1e-4)


class TestGroupedQueryAttention:


    def test_repeat_kv_heads_matches_hf_reference(self):

        torch.manual_seed(6)
        cx, cw, seq, cu = 4, 8, 5, 3
        x = torch.randn(seq, cx, cu, device="cuda")
        out = _repeat_kv_heads(x, cw)
        assert out.shape == (seq, cx * cw, cu)
        for cv in range(cx):
            for rep in range(cw):
                cy = cv * cw + rep
                torch.testing.assert_close(out[:, cy, :], x[:, cv, :])

    def test_kernel_matches_reference_at_tinyllama_shape(self):


        torch.manual_seed(7)
        dl, dk, dc, dm = 32, 4, 64, 16
        di = dl // dk
        dh = 2
        dj = dh * dm

        db = PagedKVCache(num_layers=1, num_heads=dk, head_dim=dc,
                              page_size=dm, max_pages=dh, device="cuda")
        df = torch.randn(dj, dk, dc, device="cuda")
        dv = torch.randn(dj, dk, dc, device="cuda")
        db.append(0, df, dv)
        de, du = db.k_layers[0], db.v_layers[0]
        assert not de._staging

        dn = torch.randn(dl, dc, device="cuda")
        dd = 1.0 / (dc ** 0.5)
        da = torch.tensor(de.block_table, dtype=torch.int32, device="cuda")
        dr = torch.tensor([s if s is not None else 0.0 for s in de.page_scales], device="cuda")
        ds = torch.tensor([s if s is not None else 0.0 for s in du.page_scales], device="cuda")

        dx, dw, _ = _C.paged_attention_committed(
            dn, de.storage, du.storage, dr, ds, da, dd
        )
        dg = dx / dw.unsqueeze(-1)

        do = _repeat_kv_heads(de.read_fp32(), di)
        dp = _repeat_kv_heads(du.read_fp32(), di)
        dq = _reference_full_attention(dn, do, dp, dd)

        torch.testing.assert_close(dg, dq, atol=1e-3, rtol=1e-3)

    def test_end_to_end_paged_attention_at_tinyllama_shape(self):

        torch.manual_seed(8)
        eh, eg, eb, ei = 32, 4, 64, 16
        ed, ef = 16, 5

        ec = torch.randn(ed + ef, eg, eb, device="cuda")
        eq = torch.randn(ed + ef, eg, eb, device="cuda")
        dy = PagedKVCache(num_layers=1, num_heads=eg, head_dim=eb,
                              page_size=ei, max_pages=4, device="cuda")
        dy.append(0, ec, eq)
        assert len(dy.k_layers[0].block_table) == 1
        assert len(dy.k_layers[0]._staging) == ef

        ej = torch.randn(eh, eb, device="cuda")
        got = paged_attention(dy, layer_idx=0, query=ej)
        assert got.shape == (eh, eb)

        ee = eh // eg
        ek = _repeat_kv_heads(dy.k_layers[0].read_fp32()[:ed], ee)
        el = _repeat_kv_heads(dy.v_layers[0].read_fp32()[:ed], ee)
        eo = _repeat_kv_heads(torch.stack(dy.k_layers[0]._staging), ee)
        ep = _repeat_kv_heads(torch.stack(dy.v_layers[0]._staging), ee)
        dz = torch.cat([ek, eo])
        ea = torch.cat([el, ep])
        em = _reference_full_attention(ej, dz, ea, 1.0 / (eb ** 0.5))

        torch.testing.assert_close(got, em, atol=1e-3, rtol=1e-3)
