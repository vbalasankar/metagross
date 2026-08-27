import pytest
import torch

pytest.importorskip("metagross._C", reason="metagross._C extension not built yet")

from metagross import _C
from metagross.cache import LayerKVCache, PagedKVCache
from tests.reference import quantize_symmetric_int8, dequantize_symmetric_int8


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")


def _random_tokens(d, c=4, b=8, e=3.0, f=0):
    g = torch.Generator(device="cuda").manual_seed(f)
    return (torch.randn(d, c, b, device="cuda", generator=g) * e)


class TestQuantizeKernelAgainstReference:


    def test_quantize_new_scale_matches_reference(m):
        n = _random_tokens(num_tokens=16, num_heads=4, head_dim=8)
        j = torch.zeros(16, 4, 8, dtype=torch.int8, device="cuda")

        h = _C.quantize_new_scale(n, j, 0)

        k, l = quantize_symmetric_int8(n.cpu().numpy())
        assert h == pytest.approx(l, rel=1e-5)
        torch.testing.assert_close(
            j.cpu(), torch.from_numpy(k), msg="quantized INT8 values differ from the reference"
        )

    def test_dequantize_recovers_within_half_step(s):
        t = _random_tokens(num_tokens=16, num_heads=4, head_dim=8)
        p = torch.zeros(16, 4, 8, dtype=torch.int8, device="cuda")
        r = _C.quantize_new_scale(t, p, 0)

        q = _C.dequantize_page(p, 0, 16, r)


        o = (t - q).abs().max().item()
        assert o <= r / 2 + 1e-4

    def test_fixed_scale_path_matches_reference_dequant(z):
        aa = _random_tokens(num_tokens=8, num_heads=4, head_dim=8, seed=1)
        w = torch.zeros(16, 4, 8, dtype=torch.int8, device="cuda")
        u = 0.05

        _C.quantize_fixed_scale(aa, w, 0, u)
        v = _C.dequantize_page(w, 0, 8, u)

        y = torch.clamp(torch.round(aa.cpu() / u), -127, 127).to(torch.int8)
        x = dequantize_symmetric_int8(y.numpy(), u)
        torch.testing.assert_close(v.cpu(), torch.from_numpy(x))

    def test_fixed_scale_clamps_out_of_range_values(ac):


        ad = torch.full((1, 4, 8), 1000.0, device="cuda")
        ab = torch.zeros(16, 4, 8, dtype=torch.int8, device="cuda")
        _C.quantize_fixed_scale(ad, ab, 0, 0.01)
        assert (ab[0].cpu() == 127).all()
        assert (ab[1:].cpu() == 0).all()

    def test_offset_write_does_not_disturb_earlier_tokens(aj):


        ag = torch.zeros(16, 4, 8, dtype=torch.int8, device="cuda")
        af = _random_tokens(8, seed=2)
        ai = _random_tokens(8, seed=3)

        ah = _C.quantize_new_scale(af, ag, 0)
        ae = ag.clone()
        _C.quantize_fixed_scale(ai, ag, 8, ah)

        torch.testing.assert_close(ag[:8], ae[:8])
        assert not torch.equal(ag[8:], ae[8:])


class TestLayerKVCache:


    def test_fewer_than_page_size_tokens_stay_in_staging_uncommitted(al):
        ak = LayerKVCache(num_heads=4, head_dim=8, page_size=16, max_pages=4, device="cuda")
        ak.append(_random_tokens(10))
        assert ak.seq_len == 10
        assert ak.block_table == []
        assert len(ak._staging) == 10

    def test_commit_happens_at_exactly_page_size(an):
        am = LayerKVCache(num_heads=4, head_dim=8, page_size=16, max_pages=4, device="cuda")
        am.append(_random_tokens(20))
        assert am.seq_len == 20
        assert len(am.block_table) == 1
        assert len(am._staging) == 4
        assert am.page_scales[am.block_table[0]] is not None

    def test_read_fp32_round_trips_committed_pages_within_tolerance(aw):
        ao = LayerKVCache(num_heads=4, head_dim=8, page_size=16, max_pages=4, device="cuda")
        ax = _random_tokens(32, seed=5)
        ao.append(ax)

        au = ao.read_fp32()
        assert au.shape == ax.shape
        for ap, at in enumerate(ao.block_table):
            av = ao.page_scales[at]
            ar = slice(ap * 16, (ap + 1) * 16)
            aq = (ax[ar] - au[ar]).abs().max().item()
            assert aq <= av / 2 + 1e-3

    def test_read_fp32_staging_tail_is_exact_not_quantized(ba):
        ay = LayerKVCache(num_heads=4, head_dim=8, page_size=16, max_pages=4, device="cuda")
        bb = _random_tokens(20, seed=6)
        ay.append(bb)

        az = ay.read_fp32()


        torch.testing.assert_close(az[16:], bb[16:], atol=0.0, rtol=0.0)

    def test_different_pages_get_different_scales(bg):


        bc = LayerKVCache(num_heads=4, head_dim=8, page_size=16, max_pages=4, device="cuda")
        bh = _random_tokens(16, scale=0.1, seed=7)
        bd = _random_tokens(16, scale=50.0, seed=8)
        bc.append(bh)
        bc.append(bd)

        assert len(bc.block_table) == 2
        be = bc.page_scales[bc.block_table[0]]
        bf = bc.page_scales[bc.block_table[1]]
        assert bf > be * 10

    def test_incremental_single_token_appends_match_one_shot_first_page(bk):


        bl = _random_tokens(20, seed=9)

        bj = LayerKVCache(num_heads=4, head_dim=8, page_size=16, max_pages=4, device="cuda")
        bj.append(bl)

        bi = LayerKVCache(num_heads=4, head_dim=8, page_size=16, max_pages=4, device="cuda")
        for i in range(20):
            bi.append(bl[i:i + 1])

        assert bi.seq_len == bj.seq_len == 20
        assert bi.block_table == bj.block_table
        s1 = bj.page_scales[bj.block_table[0]]
        s2 = bi.page_scales[bi.block_table[0]]
        assert s1 == pytest.approx(s2, rel=1e-5)

    def test_page_exhaustion_raises_on_second_commit(bn):


        bm = LayerKVCache(num_heads=4, head_dim=8, page_size=16, max_pages=1, device="cuda")
        with pytest.raises(RuntimeError):
            bm.append(_random_tokens(32))


class TestPagedKVCache:


    def test_layers_are_independent(bp):
        bo = PagedKVCache(num_layers=3, num_heads=4, head_dim=8, page_size=16, max_pages=4, device="cuda")
        k0, v0 = _random_tokens(16, scale=1.0, seed=10), _random_tokens(16, scale=1.0, seed=11)
        k1, v1 = _random_tokens(16, scale=1.0, seed=12), _random_tokens(16, scale=1.0, seed=13)

        bo.append(0, k0, v0)
        bo.append(1, k1, v1)

        assert bo.k_layers[0].seq_len == 16
        assert bo.k_layers[1].seq_len == 16
        assert bo.k_layers[2].seq_len == 0


        s0 = bo.k_layers[0].page_scales[bo.k_layers[0].block_table[0]]
        s1 = bo.k_layers[1].page_scales[bo.k_layers[1].block_table[0]]
        assert s0 != s1

    def test_memory_bytes_matches_shape_arithmetic(bs):
        bq = PagedKVCache(num_layers=2, num_heads=4, head_dim=8, page_size=16, max_pages=4, device="cuda")

        br = 2 * 2 * 4 * 16 * 4 * 8
        assert bq.memory_bytes() == br
