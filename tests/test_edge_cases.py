import math

import pytest
import torch

pytest.importorskip("metagross._C", reason="metagross._C extension not built yet")

from metagross import _C
from metagross.attention import paged_attention
from metagross.cache import LayerKVCache, PagedKVCache


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")


def _reference_full_attention(query, k, v, inv_sqrt_head_dim):

    d = torch.einsum("thd,hd->ht", k, query) * inv_sqrt_head_dim
    d = d - d.max(dim=-1, keepdim=True).values
    e = torch.softmax(d, dim=-1)
    return torch.einsum("ht,thd->hd", e, v)


def _random_tokens(num_tokens, num_heads=4, head_dim=8, scale=3.0, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(num_tokens, num_heads, head_dim, device=device, generator=g) * scale


class TestNegativeOffsets:


    def test_quantize_new_scale_rejects_negative_offset(self):
        p = _random_tokens(4, num_heads=2, head_dim=4)
        n = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.quantize_new_scale(p, n, -1)
        with pytest.raises(RuntimeError):
            _C.quantize_new_scale(p, n, -100)

    def test_quantize_fixed_scale_rejects_negative_offset(self):
        t = _random_tokens(4, num_heads=2, head_dim=4)
        q = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.quantize_fixed_scale(t, q, -1, 0.1)

    def test_dequantize_page_rejects_negative_offset(self):
        u = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.dequantize_page(u, -1, 4, 0.1)

    def test_dequantize_page_rejects_negative_num_tokens(self):
        x = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.dequantize_page(x, 0, -1, 0.1)

    def test_int4_quantize_rejects_negative_offset(self):
        ab = _random_tokens(4, num_heads=2, head_dim=4)
        z = torch.zeros(16, 2, 2, dtype=torch.uint8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.quantize_int4_new_scale(ab, z, -1)

    def test_int4_dequantize_rejects_negative_offset_and_num_tokens(self):
        ac = torch.zeros(16, 2, 2, dtype=torch.uint8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.dequantize_int4_page(ac, -1, 4, 0.1)
        with pytest.raises(RuntimeError):
            _C.dequantize_int4_page(ac, 0, -1, 0.1)


class TestZeroTokenOperations:


    def test_quantize_new_scale_zero_tokens_leaves_page_untouched(self):
        ag = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        ae = ag.clone()
        af = torch.empty(0, 2, 4, device="cuda")
        ah = _C.quantize_new_scale(af, ag, 0)
        assert ah == pytest.approx(1e-8)
        torch.testing.assert_close(ag, ae)

    def test_quantize_fixed_scale_zero_tokens_leaves_page_untouched(self):
        al = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        aj = al.clone()
        ak = torch.empty(0, 2, 4, device="cuda")
        _C.quantize_fixed_scale(ak, al, 0, 0.5)
        torch.testing.assert_close(al, aj)

    def test_dequantize_page_zero_tokens_returns_correctly_shaped_empty(self):
        an = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        out = _C.dequantize_page(an, 0, 0, 0.5)
        assert out.shape == (0, 2, 4)
        assert out.dtype == torch.float32
        assert out.device.type == "cuda"

    def test_int4_quantize_zero_tokens_leaves_page_untouched(self):
        ar = torch.zeros(16, 2, 2, dtype=torch.uint8, device="cuda")
        ap = ar.clone()
        aq = torch.empty(0, 2, 4, device="cuda")
        at = _C.quantize_int4_new_scale(aq, ar, 0)
        assert at == pytest.approx(1e-8)
        torch.testing.assert_close(ar, ap)

    def test_int4_dequantize_zero_tokens_returns_correctly_shaped_empty(self):
        av = torch.zeros(16, 2, 2, dtype=torch.uint8, device="cuda")
        out = _C.dequantize_int4_page(av, 0, 0, 0.5)
        assert out.shape == (0, 2, 4)
        assert out.dtype == torch.float32

    def test_layer_cache_append_zero_tokens_is_a_noop(self):
        ax = LayerKVCache(num_heads=2, head_dim=4, page_size=16, max_pages=2, device="cuda")
        ax.append(torch.empty(0, 2, 4, device="cuda"))
        assert ax.seq_len == 0
        assert ax.block_table == []


class TestInvalidDimensions:


    def test_quantize_new_scale_rejects_wrong_input_rank(self):
        az = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        bb = torch.randn(4, 4, device="cuda")
        with pytest.raises(RuntimeError):
            _C.quantize_new_scale(bb, az, 0)

    def test_quantize_new_scale_rejects_wrong_page_storage_rank(self):
        bd = _random_tokens(4, num_heads=2, head_dim=4)
        be = torch.zeros(16, 8, dtype=torch.int8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.quantize_new_scale(bd, be, 0)

    def test_quantize_new_scale_rejects_num_heads_mismatch(self):
        bh = _random_tokens(4, num_heads=2, head_dim=4)
        bf = torch.zeros(16, 3, 4, dtype=torch.int8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.quantize_new_scale(bh, bf, 0)

    def test_quantize_new_scale_rejects_head_dim_mismatch(self):
        bk = _random_tokens(4, num_heads=2, head_dim=4)
        bi = torch.zeros(16, 2, 5, dtype=torch.int8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.quantize_new_scale(bk, bi, 0)

    def test_quantize_new_scale_rejects_oversized_token_count(self):
        bn = _random_tokens(20, num_heads=2, head_dim=4)
        bl = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.quantize_new_scale(bn, bl, 0)

    def test_int4_quantize_rejects_odd_head_dim(self):
        bq = torch.randn(4, 2, 5, device="cuda")
        bo = torch.zeros(16, 2, 3, dtype=torch.uint8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.quantize_int4_new_scale(bq, bo, 0)

    def test_int4_quantize_rejects_zero_head_dim(self):
        bt = torch.randn(4, 2, 0, device="cuda")
        br = torch.zeros(16, 2, 0, dtype=torch.uint8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.quantize_int4_new_scale(bt, br, 0)

    def test_int4_quantize_rejects_wrong_packed_dimension(self):
        bv = _random_tokens(4, num_heads=2, head_dim=8)
        bw = torch.zeros(16, 2, 3, dtype=torch.uint8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.quantize_int4_new_scale(bv, bw, 0)

    def test_int4_quantize_rejects_wrong_num_heads(self):
        bz = _random_tokens(4, num_heads=2, head_dim=8)
        ca = torch.zeros(16, 3, 4, dtype=torch.uint8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.quantize_int4_new_scale(bz, ca, 0)

    def test_int4_quantize_one_token_full_page_and_page_plus_one(self):
        cg, cc, cb = 16, 2, 8
        for cd in (1, cg):
            cf = torch.zeros(cg, cc, cb // 2, dtype=torch.uint8, device="cuda")
            cj = _random_tokens(cd, num_heads=cc, head_dim=cb, seed=cd)
            ch = _C.quantize_int4_new_scale(cj, cf, 0)
            assert ch > 0

        cf = torch.zeros(cg, cc, cb // 2, dtype=torch.uint8, device="cuda")
        ce = _random_tokens(cg + 1, num_heads=cc, head_dim=cb, seed=99)
        with pytest.raises(RuntimeError):
            _C.quantize_int4_new_scale(ce, cf, 0)


class TestInvalidPageIds:


    def _committed_cache(self, num_heads=4, head_dim=64, page_size=16, max_pages=2):
        ck = PagedKVCache(num_layers=1, num_heads=num_heads, head_dim=head_dim,
                              page_size=page_size, max_pages=max_pages, device="cuda")
        ck.append(0, _random_tokens(page_size, num_heads, head_dim, seed=1),
                     _random_tokens(page_size, num_heads, head_dim, seed=2))
        return ck

    def _call_kernel_with_block_table(self, cache, block_table_values):
        cu, cz = cache.k_layers[0], cache.v_layers[0]
        cv = torch.randn(cu.num_heads, cu.head_dim, device="cuda")
        cq = torch.tensor(block_table_values, dtype=torch.int32, device="cuda")
        cw = torch.tensor([s if s is not None else 0.0 for s in cu.page_scales], device="cuda")
        cx = torch.tensor([s if s is not None else 0.0 for s in cz.page_scales], device="cuda")
        ct = 1.0 / (cu.head_dim ** 0.5)
        return _C.paged_attention_committed(
            cv, cu.storage, cz.storage, cw, cx, cq, ct
        )

    def test_valid_first_and_last_page(self):
        da = self._committed_cache(max_pages=2)
        dc = da.k_layers[0].block_table[0]
        out = self._call_kernel_with_block_table(da, [dc])
        assert torch.isfinite(out[0]).all()

    def test_negative_page_id_rejected(self):
        dd = self._committed_cache()
        with pytest.raises(RuntimeError):
            self._call_kernel_with_block_table(dd, [-1])

    def test_page_id_equal_to_max_pages_rejected(self):
        df = self._committed_cache(max_pages=2)
        with pytest.raises(RuntimeError):
            self._call_kernel_with_block_table(df, [2])

    def test_page_id_greater_than_max_pages_rejected(self):
        dh = self._committed_cache(max_pages=2)
        with pytest.raises(RuntimeError):
            self._call_kernel_with_block_table(dh, [999])


class TestPagedAttentionShapeValidation:
    def _valid_kernel_args(self, num_heads=4, head_dim=64, page_size=16, max_pages=2):
        dk = PagedKVCache(num_layers=1, num_heads=num_heads, head_dim=head_dim,
                              page_size=page_size, max_pages=max_pages, device="cuda")
        dk.append(0, _random_tokens(page_size, num_heads, head_dim, seed=3),
                     _random_tokens(page_size, num_heads, head_dim, seed=4))
        dm, du = dk.k_layers[0], dk.v_layers[0]
        dq = torch.randn(num_heads, head_dim, device="cuda")
        dj = torch.tensor(dm.block_table, dtype=torch.int32, device="cuda")
        dr = torch.tensor([s if s is not None else 0.0 for s in dm.page_scales], device="cuda")
        ds = torch.tensor([s if s is not None else 0.0 for s in du.page_scales], device="cuda")
        return dict(query=dq, k_storage=dm.storage, v_storage=du.storage,
                    scales_k=dr, scales_v=ds, block_table=dj,
                    inv_sqrt_d=1.0 / (head_dim ** 0.5))

    def test_valid_call_succeeds(self):
        a = self._valid_kernel_args()
        out = _C.paged_attention_committed(a["query"], a["k_storage"], a["v_storage"],
                                            a["scales_k"], a["scales_v"], a["block_table"], a["inv_sqrt_d"])
        assert torch.isfinite(out[0]).all()

    def test_query_wrong_rank_rejected(self):
        a = self._valid_kernel_args()
        dw = a["query"].reshape(-1)
        with pytest.raises(RuntimeError):
            _C.paged_attention_committed(dw, a["k_storage"], a["v_storage"],
                                          a["scales_k"], a["scales_v"], a["block_table"], a["inv_sqrt_d"])

    def test_query_wrong_head_dim_rejected(self):
        a = self._valid_kernel_args()
        dy = torch.randn(a["query"].shape[0], 32, device="cuda")
        with pytest.raises(RuntimeError):
            _C.paged_attention_committed(dy, a["k_storage"], a["v_storage"],
                                          a["scales_k"], a["scales_v"], a["block_table"], a["inv_sqrt_d"])

    def test_k_v_page_count_mismatch_rejected(self):
        a = self._valid_kernel_args()
        ea = a["v_storage"][:1]
        with pytest.raises(RuntimeError):
            _C.paged_attention_committed(a["query"], a["k_storage"], ea,
                                          a["scales_k"], a["scales_v"], a["block_table"], a["inv_sqrt_d"])

    def test_k_v_head_count_mismatch_rejected(self):
        a = self._valid_kernel_args(num_heads=4)

        ec = torch.zeros(a["v_storage"].shape[0], a["v_storage"].shape[1], 8,
                                     a["v_storage"].shape[3], dtype=torch.int8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.paged_attention_committed(a["query"], a["k_storage"], ec,
                                          a["scales_k"], a["scales_v"], a["block_table"], a["inv_sqrt_d"])

    def test_k_v_page_size_mismatch_rejected(self):
        a = self._valid_kernel_args(page_size=16)
        ee = torch.zeros(a["v_storage"].shape[0], 4, a["v_storage"].shape[2],
                                     a["v_storage"].shape[3], dtype=torch.int8, device="cuda")
        with pytest.raises(RuntimeError):
            _C.paged_attention_committed(a["query"], a["k_storage"], ee,
                                          a["scales_k"], a["scales_v"], a["block_table"], a["inv_sqrt_d"])

    def test_page_size_exceeding_kernel_max_rejected(self):

        ek, ei, el = 4, 64, 128
        eh = PagedKVCache(num_layers=1, num_heads=ek, head_dim=ei,
                              page_size=el, max_pages=1, device="cuda")
        eh.append(0, _random_tokens(el, ek, ei, seed=5),
                     _random_tokens(el, ek, ei, seed=6))
        ej, eq = eh.k_layers[0], eh.v_layers[0]
        em = torch.randn(ek, ei, device="cuda")
        eg = torch.tensor(ej.block_table, dtype=torch.int32, device="cuda")
        en = torch.tensor([s if s is not None else 0.0 for s in ej.page_scales], device="cuda")
        eo = torch.tensor([s if s is not None else 0.0 for s in eq.page_scales], device="cuda")
        with pytest.raises(RuntimeError):
            _C.paged_attention_committed(em, ej.storage, eq.storage,
                                          en, eo, eg, 1.0 / (ei ** 0.5))

    def test_runtime_page_size_4_and_16_both_supported(self):


        for es in (4, 16):
            a = self._valid_kernel_args(page_size=es, max_pages=2)
            out = _C.paged_attention_committed(a["query"], a["k_storage"], a["v_storage"],
                                                a["scales_k"], a["scales_v"], a["block_table"], a["inv_sqrt_d"])
            ev, eu, er = out
            assert torch.isfinite(ev).all()
            assert torch.isfinite(eu).all()
            assert (eu > 0).all()


class TestEmptyAttentionCache:
    def test_paged_attention_on_never_appended_cache_raises_value_error(self):
        ew = PagedKVCache(num_layers=1, num_heads=4, head_dim=64, page_size=16, max_pages=2, device="cuda")
        ex = torch.randn(4, 64, device="cuda")
        with pytest.raises(ValueError):
            paged_attention(ew, layer_idx=0, query=ex)


class TestInvalidScale:


    def _committed_cache(self):
        ez = PagedKVCache(num_layers=1, num_heads=4, head_dim=64, page_size=16, max_pages=2, device="cuda")
        ez.append(0, _random_tokens(16, 4, 64, seed=7), _random_tokens(16, 4, 64, seed=8))
        return ez

    @pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
    def test_paged_attention_rejects_invalid_scale(self, bad_scale):
        fc = self._committed_cache()
        fd = torch.randn(4, 64, device="cuda")
        with pytest.raises(ValueError):
            paged_attention(fc, layer_idx=0, query=fd, scale=bad_scale)

    @pytest.mark.parametrize("good_scale", [1.0, 0.125, 1e-6, 1e6])
    def test_paged_attention_accepts_valid_scale(self, good_scale):
        ff = self._committed_cache()
        fh = torch.randn(4, 64, device="cuda")
        out = paged_attention(ff, layer_idx=0, query=fh, scale=good_scale)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
    def test_raw_kernel_rejects_invalid_scale(self, bad_scale):
        fl = self._committed_cache()
        fm, fr = fl.k_layers[0], fl.v_layers[0]
        fn = torch.randn(4, 64, device="cuda")
        fk = torch.tensor(fm.block_table, dtype=torch.int32, device="cuda")
        fo = torch.tensor([s if s is not None else 0.0 for s in fm.page_scales], device="cuda")
        fp = torch.tensor([s if s is not None else 0.0 for s in fr.page_scales], device="cuda")
        with pytest.raises(RuntimeError):
            _C.paged_attention_committed(fn, fm.storage, fr.storage,
                                          fo, fp, fk, bad_scale)


class TestCacheConstructorValidation:
    @pytest.mark.parametrize("num_heads", [0, -1])
    def test_layer_cache_rejects_bad_num_heads(self, num_heads):
        with pytest.raises(ValueError):
            LayerKVCache(num_heads=num_heads, head_dim=8, page_size=16, max_pages=2, device="cuda")

    @pytest.mark.parametrize("head_dim", [0, -1])
    def test_layer_cache_rejects_bad_head_dim(self, head_dim):
        with pytest.raises(ValueError):
            LayerKVCache(num_heads=4, head_dim=head_dim, page_size=16, max_pages=2, device="cuda")

    @pytest.mark.parametrize("page_size", [0, -1])
    def test_layer_cache_rejects_bad_page_size(self, page_size):
        with pytest.raises(ValueError):
            LayerKVCache(num_heads=4, head_dim=8, page_size=page_size, max_pages=2, device="cuda")

    @pytest.mark.parametrize("max_pages", [0, -1])
    def test_layer_cache_rejects_bad_max_pages(self, max_pages):
        with pytest.raises(ValueError):
            LayerKVCache(num_heads=4, head_dim=8, page_size=16, max_pages=max_pages, device="cuda")

    @pytest.mark.parametrize("num_layers", [0, -1])
    def test_paged_cache_rejects_bad_num_layers(self, num_layers):
        with pytest.raises(ValueError):
            PagedKVCache(num_layers=num_layers, num_heads=4, head_dim=8, page_size=16, max_pages=2, device="cuda")


class TestKVMismatch:
    def _cache(self):
        return PagedKVCache(num_layers=1, num_heads=4, head_dim=8, page_size=16, max_pages=2, device="cuda")

    def test_token_count_mismatch_rejected(self):
        gd = self._cache()
        k = _random_tokens(5, 4, 8, seed=9)
        v = _random_tokens(6, 4, 8, seed=10)
        with pytest.raises(ValueError):
            gd.append(0, k, v)

    def test_trailing_shape_mismatch_rejected(self):
        gf = self._cache()
        k = _random_tokens(5, num_heads=4, head_dim=8, seed=11)
        v = _random_tokens(5, num_heads=3, head_dim=8, seed=12)
        with pytest.raises(ValueError):
            gf.append(0, k, v)

    def test_wrong_rank_rejected(self):
        gh = self._cache()
        k = torch.randn(5, 4, 8, device="cuda")
        v = torch.randn(5, 32, device="cuda")
        with pytest.raises(ValueError):
            gh.append(0, k, v)

    def test_wrong_head_count_rejected(self):
        gj = self._cache()
        k = _random_tokens(5, num_heads=99, head_dim=8, seed=13)
        v = _random_tokens(5, num_heads=99, head_dim=8, seed=14)
        with pytest.raises(ValueError):
            gj.append(0, k, v)

    def test_wrong_head_dim_rejected(self):
        gl = self._cache()
        k = _random_tokens(5, num_heads=4, head_dim=99, seed=15)
        v = _random_tokens(5, num_heads=4, head_dim=99, seed=16)
        with pytest.raises(ValueError):
            gl.append(0, k, v)

    def test_layer_idx_out_of_range_rejected(self):
        gn = self._cache()
        k = _random_tokens(5, 4, 8, seed=17)
        v = _random_tokens(5, 4, 8, seed=18)
        with pytest.raises(ValueError):
            gn.append(5, k, v)
        with pytest.raises(ValueError):
            gn.append(-1, k, v)


class TestPageBoundaryTransitions:


    @pytest.mark.parametrize("page_size", [4, 16])
    @pytest.mark.parametrize("seq_len", [0, 1, 2, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65])
    def test_seq_len_boundary(self, page_size, seq_len):
        if seq_len == 0:
            pytest.skip("0-token append is covered by TestZeroTokenOperations")
        gs = seq_len // page_size + 2
        gp = LayerKVCache(num_heads=2, head_dim=8, page_size=page_size, max_pages=gs, device="cuda")
        gp.append(_random_tokens(seq_len, num_heads=2, head_dim=8, seed=seq_len))

        gq = seq_len // page_size
        gr = seq_len % page_size

        assert gp.seq_len == seq_len
        assert len(gp.block_table) == gq
        assert len(gp._staging) == gr

        gu = gp.read_fp32()
        assert gu.shape[0] == seq_len


class TestAttentionAtRuntimePageSize4:


    def test_committed_only(self):
        hb, gy, hc = 4, 64, 4
        ha = 2 * hc
        gx = PagedKVCache(num_layers=1, num_heads=hb, head_dim=gy,
                              page_size=hc, max_pages=4, device="cuda")
        gz = _random_tokens(ha, hb, gy, seed=20)
        hj = _random_tokens(ha, hb, gy, seed=21)
        gx.append(0, gz, hj)
        assert not gx.k_layers[0]._staging

        he = torch.randn(hb, gy, device="cuda")
        got = paged_attention(gx, layer_idx=0, query=he)

        hf = gx.k_layers[0].read_fp32()
        hg = gx.v_layers[0].read_fp32()
        hh = _reference_full_attention(he, hf, hg, 1.0 / (gy ** 0.5))
        torch.testing.assert_close(got, hh, atol=1e-3, rtol=1e-3)

    def test_staging_only(self):
        ho, hl, hp = 4, 64, 4
        hn = hp - 1
        hk = PagedKVCache(num_layers=1, num_heads=ho, head_dim=hl,
                              page_size=hp, max_pages=4, device="cuda")
        hm = _random_tokens(hn, ho, hl, seed=22)
        hu = _random_tokens(hn, ho, hl, seed=23)
        hk.append(0, hm, hu)
        assert hk.k_layers[0].block_table == []

        hq = torch.randn(ho, hl, device="cuda")
        got = paged_attention(hk, layer_idx=0, query=hq)
        hr = _reference_full_attention(hq, hm, hu, 1.0 / (hl ** 0.5))
        torch.testing.assert_close(got, hr, atol=1e-4, rtol=1e-4)

    def test_mixed_committed_and_staging(self):
        ic, hy, id = 4, 64, 4
        ia, ib = id, 3
        hv = PagedKVCache(num_layers=1, num_heads=ic, head_dim=hy,
                              page_size=id, max_pages=4, device="cuda")
        hz = _random_tokens(ia + ib, ic, hy, seed=24)
        im = _random_tokens(ia + ib, ic, hy, seed=25)
        hv.append(0, hz, im)
        assert len(hv.k_layers[0].block_table) == 1
        assert len(hv.k_layers[0]._staging) == ib

        ie = torch.randn(ic, hy, device="cuda")
        got = paged_attention(hv, layer_idx=0, query=ie)

        ig = hv.k_layers[0].read_fp32()[:ia]
        ih = hv.v_layers[0].read_fp32()[:ia]
        ik = torch.stack(hv.k_layers[0]._staging)
        il = torch.stack(hv.v_layers[0]._staging)
        hw = torch.cat([ig, ik])
        hx = torch.cat([ih, il])
        ii = _reference_full_attention(ie, hw, hx, 1.0 / (hy ** 0.5))

        torch.testing.assert_close(got, ii, atol=1e-3, rtol=1e-3)


class TestGQARatios:
    @pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 8), (8, 4), (16, 4), (32, 4), (32, 8)])
    def test_valid_ratio_succeeds(self, num_q_heads, num_kv_heads):
        ip, iu = 64, 16
        io = PagedKVCache(num_layers=1, num_heads=num_kv_heads, head_dim=ip,
                              page_size=iu, max_pages=2, device="cuda")
        iq = _random_tokens(iu, num_kv_heads, ip, seed=num_q_heads + num_kv_heads)
        ix = _random_tokens(iu, num_kv_heads, ip, seed=num_q_heads - num_kv_heads)
        io.append(0, iq, ix)
        iv = torch.randn(num_q_heads, ip, device="cuda")
        out = paged_attention(io, layer_idx=0, query=iv)
        assert out.shape == (num_q_heads, ip)
        assert torch.isfinite(out).all()

    def test_invalid_ratio_rejected_by_wrapper(self):

        iz, jb, ja = 64, 16, 4
        iy = PagedKVCache(num_layers=1, num_heads=ja, head_dim=iz,
                              page_size=jb, max_pages=2, device="cuda")
        iy.append(0, _random_tokens(jb, ja, iz, seed=30),
                     _random_tokens(jb, ja, iz, seed=31))
        jc = torch.randn(7, iz, device="cuda")
        with pytest.raises(ValueError):
            paged_attention(iy, layer_idx=0, query=jc)

    def test_invalid_ratio_rejected_by_raw_kernel(self):
        jg, jj, ji = 64, 16, 4
        jf = PagedKVCache(num_layers=1, num_heads=ji, head_dim=jg,
                              page_size=jj, max_pages=2, device="cuda")
        jf.append(0, _random_tokens(jj, ji, jg, seed=32),
                     _random_tokens(jj, ji, jg, seed=33))
        jh, jo = jf.k_layers[0], jf.v_layers[0]
        jk = torch.randn(7, jg, device="cuda")
        je = torch.tensor(jh.block_table, dtype=torch.int32, device="cuda")
        jl = torch.tensor([s if s is not None else 0.0 for s in jh.page_scales], device="cuda")
        jm = torch.tensor([s if s is not None else 0.0 for s in jo.page_scales], device="cuda")
        with pytest.raises(RuntimeError):
            _C.paged_attention_committed(jk, jh.storage, jo.storage,
                                          jl, jm, je, 1.0 / (jg ** 0.5))


class TestNonContiguousInputs:


    def test_quantize_input_supports_non_contiguous(self):


        js, jr, jq = 4, 4, 8
        jp = torch.randn(jq, jr, js, device="cuda")
        jw = jp.permute(2, 1, 0)
        assert not jw.is_contiguous()
        assert jw.shape == (js, jr, jq)
        jt = torch.zeros(16, jr, jq, dtype=torch.int8, device="cuda")
        ju = _C.quantize_new_scale(jw, jt, 0)
        assert ju > 0

    def test_quantize_page_storage_rejects_non_contiguous(self):
        jx = torch.zeros(8, 4, 16, dtype=torch.int8, device="cuda")
        jz = jx.permute(2, 1, 0)
        assert not jz.is_contiguous()
        ka = _random_tokens(4, num_heads=4, head_dim=8, seed=40)
        with pytest.raises(RuntimeError):
            _C.quantize_new_scale(ka, jz, 0)

    def test_paged_attention_query_supports_non_contiguous(self):
        ke, kd, kf = 4, 64, 16
        kc = PagedKVCache(num_layers=1, num_heads=ke, head_dim=kd,
                              page_size=kf, max_pages=2, device="cuda")
        kc.append(0, _random_tokens(kf, ke, kd, seed=41),
                     _random_tokens(kf, ke, kd, seed=42))
        kb = torch.randn(kd, ke, device="cuda")
        kh = kb.transpose(0, 1)
        assert not kh.is_contiguous()
        out = paged_attention(kc, layer_idx=0, query=kh)
        assert torch.isfinite(out).all()

    def test_paged_attention_k_storage_rejects_non_contiguous(self):
        km, kk, ko = 4, 64, 16
        kj = PagedKVCache(num_layers=1, num_heads=km, head_dim=kk,
                              page_size=ko, max_pages=2, device="cuda")
        kj.append(0, _random_tokens(ko, km, kk, seed=43),
                     _random_tokens(ko, km, kk, seed=44))
        kl, ku = kj.k_layers[0], kj.v_layers[0]


        kn = torch.zeros(*kl.storage.shape[:-1], kk + 1, dtype=torch.int8, device="cuda")
        kn[..., :kk] = kl.storage
        kt = kn[..., :kk]
        assert not kt.is_contiguous()
        assert kt.shape == kl.storage.shape
        kp = torch.randn(km, kk, device="cuda")
        ki = torch.tensor(kl.block_table, dtype=torch.int32, device="cuda")
        kq = torch.tensor([s if s is not None else 0.0 for s in kl.page_scales], device="cuda")
        kr = torch.tensor([s if s is not None else 0.0 for s in ku.page_scales], device="cuda")
        with pytest.raises(RuntimeError):
            _C.paged_attention_committed(kp, kt, ku.storage,
                                          kq, kr, ki, 1.0 / (kk ** 0.5))


class TestNumericalEdgeCases:


    def test_all_zeros(self):
        ky = torch.zeros(4, 2, 4, device="cuda")
        kv = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        kw = _C.quantize_new_scale(ky, kv, 0)
        assert kw == pytest.approx(1e-8)
        assert (kv[:4] == 0).all()

    def test_all_ones(self):
        ld = torch.ones(4, 2, 4, device="cuda")
        kz = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        lb = _C.quantize_new_scale(ld, kz, 0)
        la = _C.dequantize_page(kz, 0, 4, lb)
        torch.testing.assert_close(la, ld, atol=lb / 2 + 1e-4, rtol=0)

    def test_all_negative(self):
        lh = -torch.ones(4, 2, 4, device="cuda") * 5.0
        le = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        lf = _C.quantize_new_scale(lh, le, 0)
        assert (le[:4].to(torch.int32) <= 0).all()

    def test_values_beyond_int8_range_clamp_to_127(self):
        lk = torch.full((1, 2, 4), 1e6, device="cuda")
        li = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        _C.quantize_fixed_scale(lk, li, 0, 0.01)
        assert (li[0].to(torch.int32) == 127).all()

    def test_negative_values_beyond_int8_range_clamp_to_neg_127(self):
        ln = torch.full((1, 2, 4), -1e6, device="cuda")
        ll = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        _C.quantize_fixed_scale(ln, ll, 0, 0.01)
        assert (ll[0].to(torch.int32) == -127).all()

    def test_extreme_finite_value_with_tiny_scale_does_not_crash(self):


        lr = torch.full((1, 2, 4), 1e30, device="cuda")
        lo = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        _C.quantize_fixed_scale(lr, lo, 0, 1e-10)
        lp = lo[0].to(torch.int32)
        assert (lp >= -127).all() and (lp <= 127).all()

    def test_nan_input_does_not_crash_and_stays_in_range(self):
        lv = torch.full((1, 2, 4), float("nan"), device="cuda")
        ls = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        _C.quantize_fixed_scale(lv, ls, 0, 0.1)
        lt = ls[0].to(torch.int32)
        assert (lt >= -127).all() and (lt <= 127).all()
        assert torch.isfinite(lt.to(torch.float32)).all()

    @pytest.mark.parametrize("inf_value", [float("inf"), float("-inf")])
    def test_inf_input_does_not_crash_and_stays_in_range(self, inf_value):
        ma = torch.full((1, 2, 4), inf_value, device="cuda")
        lx = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        _C.quantize_fixed_scale(ma, lx, 0, 0.1)
        ly = lx[0].to(torch.int32)
        assert (ly >= -127).all() and (ly <= 127).all()

    def test_new_scale_with_inf_in_tensor_does_not_crash(self):


        mf = _random_tokens(4, num_heads=2, head_dim=4, seed=50)
        mf[0, 0, 0] = float("inf")
        mb = torch.zeros(16, 2, 4, dtype=torch.int8, device="cuda")
        md = _C.quantize_new_scale(mf, mb, 0)
        assert math.isfinite(md) and md > 0
        mc = _C.dequantize_page(mb, 0, 4, md)
        assert torch.isfinite(mc).all()
