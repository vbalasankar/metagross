import pytest
import torch

pytest.importorskip("metagross._C", reason="metagross._C extension not built yet")

from metagross import _C
from metagross.attention import paged_attention
from metagross.cache import LayerKVCache, PagedKVCache


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")

BASE_SEED = 20260822
N_TRIALS = 25


def _rng(b):


    return torch.Generator().manual_seed(BASE_SEED + b)


def _randn_cuda(g, *d, c=1.0):
    return (torch.randn(*d, generator=g) * c).cuda()


def _randint(g, low, e):
    return int(torch.randint(low, e, (1,), generator=g).item())


def _reference_full_attention(l, k, v, h):

    m = torch.einsum("thd,hd->ht", k, l) * h
    m = m - m.max(dim=-1, keepdim=True).values
    o = torch.softmax(m, dim=-1)
    return torch.einsum("ht,thd->hd", o, v)


class TestRandomizedInt8:
    def test_quantize_dequantize_roundtrip(aa):
        for ab in range(N_TRIALS):
            g = _rng(ab)
            w = _randint(g, 1, 17)
            u = _randint(g, 1, 9)
            q = _randint(g, 1, 33)
            r = torch.rand((1,), generator=g).item() * 10 + 0.01

            ac = _randn_cuda(g, w, u, q, z=r)
            x = torch.zeros(w, u, q, dtype=torch.int8, device="cuda")
            z = _C.quantize_new_scale(ac, x, 0)
            y = _C.dequantize_page(x, 0, w, z)

            assert torch.isfinite(y).all(), f"trial {ab}: non-finite recovered value"


            t = (y - ac).abs().max().item()
            assert t <= z / 2 + 1e-4, f"trial {ab}: error {t} exceeds scale/2={z / 2}"


class TestRandomizedInt4:
    def test_quantize_dequantize_roundtrip(al):
        for am in range(N_TRIALS):
            g = _rng(1000 + am)
            ah = _randint(g, 1, 17)
            ag = _randint(g, 1, 9)
            ad = _randint(g, 1, 17) * 2
            ae = torch.rand((1,), generator=g).item() * 10 + 0.01

            ao = _randn_cuda(g, ah, ag, ad, ak=ae)
            ai = torch.zeros(ah, ag, ad // 2, dtype=torch.uint8, device="cuda")
            ak = _C.quantize_int4_new_scale(ao, ai, 0)
            aj = _C.dequantize_int4_page(ai, 0, ah, ak)

            assert torch.isfinite(aj).all(), f"trial {am}: non-finite recovered value"
            af = (aj - ao).abs().max().item()
            assert af <= ak / 2 + 1e-4, f"trial {am}: error {af} exceeds scale/2={ak / 2}"


class TestRandomizedPageTransitions:
    def test_random_seq_len_and_page_size(ax):
        for az in range(N_TRIALS):
            g = _rng(2000 + az)
            av = _randint(g, 1, 33)
            ay = _randint(g, 0, 4 * av + 1)
            au = _randint(g, 1, 5)
            aq = _randint(g, 1, 17)
            ar = ay // av + 2

            ap = LayerKVCache(au=au, aq=aq, av=av,
                                  ar=ar, device="cuda")
            ba = _randn_cuda(g, ay, au, aq, scale=3.0)
            ap.append(ba)

            assert ap.seq_len == ay, f"trial {az}: seq_len mismatch"
            assert len(ap.block_table) == ay // av, f"trial {az}: committed page count mismatch"
            assert len(ap._staging) == ay % av, f"trial {az}: staging length mismatch"

            aw = ap.read_fp32()
            assert aw.shape[0] == ay, f"trial {az}: read_fp32 length mismatch"
            assert torch.isfinite(aw).all(), f"trial {az}: non-finite recovered value"


class TestRandomizedAttentionShapes:
    def test_random_shapes_match_reference(bo):
        for bq in range(N_TRIALS):
            g = _rng(3000 + bq)


            bc = 64
            bi = _randint(g, 1, 5)
            bj = _randint(g, 1, 17)
            bg = _randint(g, 0, 4)
            bh = _randint(g, 0, bj)
            bp = bg * bj + bh
            if bp == 0:
                continue
            bf = bg + 2

            bb = PagedKVCache(num_layers=1, num_heads=bi, bc=bc,
                                  bj=bj, bf=bf, device="cuda")
            bd = _randn_cuda(g, bp, bi, bc, scale=3.0)
            br = _randn_cuda(g, bp, bi, bc, scale=3.0)
            bb.append(0, bd, br)

            bk = _randn_cuda(g, bi, bc)
            got = paged_attention(bb, layer_idx=0, bk=bk)

            bl, bm = bb.read_layer_fp32(0)
            bn = _reference_full_attention(bk, bl, bm, 1.0 / (bc ** 0.5))

            assert torch.isfinite(got).all(), f"trial {bq}: non-finite output"
            assert torch.allclose(got, bn, atol=5e-3, rtol=5e-3), (
                f"trial {bq} (page_size={bj}, n_committed_pages={bg}, "
                f"n_staged={bh}, num_kv_heads={bi}): "
                f"max diff {(got - bn).abs().max().item()}"
            )


class TestRandomizedGQA:
    def test_random_valid_gqa_ratios_match_reference(cg):
        bt = 64
        ck = [1, 2, 4, 8]
        for ci in range(N_TRIALS):
            g = _rng(4000 + ci)
            bw = ck[_randint(g, 0, len(ck))]
            bv = _randint(g, 1, 9)
            bx = bw * bv
            bz = 16
            ch = _randint(g, 1, 3) * bz

            bs = PagedKVCache(num_layers=1, num_heads=bw, bt=bt,
                                  bz=bz, max_pages=ch // bz + 1, device="cuda")
            bu = _randn_cuda(g, ch, bw, bt, scale=3.0)
            cj = _randn_cuda(g, ch, bw, bt, scale=3.0)
            bs.append(0, bu, cj)

            ca = _randn_cuda(g, bx, bt)
            got = paged_attention(bs, layer_idx=0, ca=ca)

            assert got.shape == (bx, bt), f"trial {ci}: output shape mismatch"
            assert torch.isfinite(got).all(), f"trial {ci}: non-finite output"


            cb, cc = bs.read_layer_fp32(0)
            ce = cb.repeat_interleave(bv, dim=1)
            cf = cc.repeat_interleave(bv, dim=1)
            cd = _reference_full_attention(ca, ce, cf, 1.0 / (bt ** 0.5))
            assert torch.allclose(got, cd, atol=5e-3, rtol=5e-3), (
                f"trial {ci} (num_kv_heads={bw}, n_rep={bv}): "
                f"max diff {(got - cd).abs().max().item()}"
            )

    def test_random_invalid_gqa_ratios_rejected(cs):
        cn, cq, co = 64, 16, 4
        for ct in range(N_TRIALS):
            g = _rng(5000 + ct)


            cm = _randint(g, 1, 40)
            if cm % co == 0:
                cm += 1
            cp = cm

            cl = PagedKVCache(num_layers=1, num_heads=co, cn=cn,
                                  cq=cq, max_pages=2, device="cuda")
            cl.append(0, _randn_cuda(g, cq, co, cn),
                         _randn_cuda(g, cq, co, cn))
            cr = _randn_cuda(g, cp, cn)
            with pytest.raises(ValueError):
                paged_attention(cl, layer_idx=0, cr=cr)


class TestRandomizedStagingLengths:
    def test_staging_only_matches_reference_at_random_lengths(dc):
        cv, cy, cz = 64, 4, 16
        for dd in range(N_TRIALS):
            g = _rng(6000 + dd)
            cx = _randint(g, 1, cz)

            cu = PagedKVCache(num_layers=1, cy=cy, cv=cv,
                                  cz=cz, max_pages=2, device="cuda")
            cw = _randn_cuda(g, cx, cy, cv, scale=3.0)
            de = _randn_cuda(g, cx, cy, cv, scale=3.0)
            cu.append(0, cw, de)
            assert cu.k_layers[0].block_table == [], f"trial {dd}: expected nothing committed"

            da = _randn_cuda(g, cy, cv)
            got = paged_attention(cu, layer_idx=0, da=da)
            db = _reference_full_attention(da, cw, de, 1.0 / (cv ** 0.5))
            assert torch.allclose(got, db, atol=1e-3, rtol=1e-3), (
                f"trial {dd} (n_staged={cx}): max diff {(got - db).abs().max().item()}"
            )


class TestRandomizedBlockTables:
    def test_shuffled_logical_order_matches_reference(ea):


        dj, dq, dr, dp = 64, 4, 16, 4
        for ec in range(N_TRIALS):
            g = _rng(7000 + ec)
            dg = PagedKVCache(num_layers=1, dq=dq, dj=dj,
                                  dr=dr, max_pages=dp, device="cuda")
            for p in range(dp):
                k = _randn_cuda(g, dr, dq, dj, scale=3.0)
                v = _randn_cuda(g, dr, dq, dj, scale=3.0)
                dg.append(0, k, v)

            dl, ed = dg.k_layers[0], dg.v_layers[0]
            dt = list(dl.block_table)
            ds = torch.randperm(dp, generator=g).tolist()
            eb = [dt[i] for i in ds]

            du = _randn_cuda(g, dq, dj)
            df = torch.tensor(eb, dtype=torch.int32, device="cuda")
            dy = torch.tensor([s if s is not None else 0.0 for s in dl.page_scales], device="cuda")
            dz = torch.tensor([s if s is not None else 0.0 for s in ed.page_scales], device="cuda")
            dk = 1.0 / (dj ** 0.5)
            eg, ef, _ = _C.paged_attention_committed(
                du, dl.storage, ed.storage, dy, dz, df, dk
            )
            got = eg / ef.unsqueeze(-1)


            dh = {
                p: _C.dequantize_page(dl.storage[p], 0, dr, dl.page_scales[p])
                for p in dt
            }
            di = {
                p: _C.dequantize_page(ed.storage[p], 0, dr, ed.page_scales[p])
                for p in dt
            }
            dv = torch.cat([dh[dt[i]] for i in ds], dim=0)
            dw = torch.cat([di[dt[i]] for i in ds], dim=0)
            dx = _reference_full_attention(du, dv, dw, dk)

            dn = (got - dx).abs().max().item()
            do = (got - dx).abs().mean().item()
            dm = [dl.page_scales[p] for p in dt]
            ee = [ed.page_scales[p] for p in dt]
            print(
                f"\ntrial {ec}: page_size={dr}, physical_pages(commit order)={dt}, "
                f"permutation={ds}, shuffled_block_table={eb}\n"
                f"  K scales (commit order)={[f'{s:.5f}' for s in dm]}\n"
                f"  V scales (commit order)={[f'{s:.5f}' for s in ee]}\n"
                f"  max_abs_diff={dn:.6f}, mean_abs_diff={do:.6f}"
            )

            assert torch.allclose(got, dx, atol=5e-3, rtol=5e-3), (
                f"trial {ec}: max diff {dn} exceeds tolerance even against the DEQUANTIZED "
                f"reference -- unlike quantization error (expected, bounded by ~scale/2 per element), this "
                f"would point at a real indexing bug. See printed diagnostics above."
            )

    def test_shuffled_logical_order_exact_under_zero_quantization_error(fc):


        ej, eo, eq, en = 64, 4, 16, 4
        for fe in range(5):
            g = _rng(7100 + fe)
            ei = PagedKVCache(num_layers=1, eo=eo, ej=ej,
                                  eq=eq, max_pages=en, device="cuda")
            ep = []
            er = []
            for p in range(en):
                k = torch.randint(-127, 128, (eq, eo, ej), generator=g).float().cuda()
                v = torch.randint(-127, 128, (eq, eo, ej), generator=g).float().cuda()
                k[0, 0, 0] = 127.0
                v[0, 0, 0] = 127.0
                ei.append(0, k, v)
                ep.append(k)
                er.append(v)

            el, ff = ei.k_layers[0], ei.v_layers[0]
            et = list(el.block_table)


            for j, p in enumerate(et):
                assert el.page_scales[p] == pytest.approx(1.0, abs=1e-6), (
                    f"trial {fe} page {j}: expected scale==1.0 (exact quantization premise violated), "
                    f"got {el.page_scales[p]}"
                )
                ev = _C.dequantize_page(el.storage[p], 0, eq, el.page_scales[p])
                torch.testing.assert_close(ev, ep[j], atol=0, rtol=0)
                ew = _C.dequantize_page(ff.storage[p], 0, eq, ff.page_scales[p])
                torch.testing.assert_close(ew, er[j], atol=0, rtol=0)

            es = torch.randperm(en, generator=g).tolist()
            fd = [et[i] for i in es]

            eu = _randn_cuda(g, eo, ej)
            eh = torch.tensor(fd, dtype=torch.int32, device="cuda")
            fa = torch.tensor([s if s is not None else 0.0 for s in el.page_scales], device="cuda")
            fb = torch.tensor([s if s is not None else 0.0 for s in ff.page_scales], device="cuda")
            ek = 1.0 / (ej ** 0.5)
            fh, fg, _ = _C.paged_attention_committed(
                eu, el.storage, ff.storage, fa, fb, eh, ek
            )
            got = fh / fg.unsqueeze(-1)


            ex = torch.cat([ep[i] for i in es], dim=0)
            ey = torch.cat([er[i] for i in es], dim=0)
            ez = _reference_full_attention(eu, ex, ey, ek)

            em = (got - ez).abs().max().item()


            assert torch.allclose(got, ez, atol=1e-3, rtol=1e-3), (
                f"trial {fe}: max diff {em} under EXACT (zero-error) quantization -- "
                f"this can only be an indexing or computation bug, not quantization noise "
                f"(permutation={es}, shuffled_block_table={fd})"
            )
