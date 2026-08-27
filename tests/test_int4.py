import pytest
import torch

pytest.importorskip("metagross._C", reason="metagross._C extension not built yet")

from metagross import _C
from tests.reference import dequantize_symmetric_int4_packed, quantize_symmetric_int4_packed


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")


def _random_tokens(num_tokens, num_heads=4, head_dim=8, scale=3.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(num_tokens, num_heads, head_dim, device="cuda", generator=g) * scale


class TestInt4KernelAgainstReference:
    def test_quantize_matches_reference(self):
        n = _random_tokens(num_tokens=16, num_heads=4, head_dim=8)
        j = torch.zeros(16, 4, 4, dtype=torch.uint8, device="cuda")

        i = _C.quantize_int4_new_scale(n, j, 0)

        h = n.cpu().numpy().reshape(-1)
        k, l = quantize_symmetric_int4_packed(h)
        assert i == pytest.approx(l, rel=1e-5)
        torch.testing.assert_close(
            j.cpu().numpy().reshape(-1),
            k,
            msg="packed INT4 bytes differ from the reference",
        )

    def test_dequantize_recovers_within_half_step(self):
        t = _random_tokens(num_tokens=16, num_heads=4, head_dim=8)
        p = torch.zeros(16, 4, 4, dtype=torch.uint8, device="cuda")
        r = _C.quantize_int4_new_scale(t, p, 0)

        q = _C.dequantize_int4_page(p, 0, 16, r)

        o = (t - q).abs().max().item()
        assert o <= r / 2 + 1e-3

    def test_roundtrip_matches_reference_exactly(self):

        ac = _random_tokens(num_tokens=8, num_heads=2, head_dim=8, seed=3)
        w = torch.zeros(8, 2, 4, dtype=torch.uint8, device="cuda")

        aa = _C.quantize_int4_new_scale(ac, w, 0)
        v = _C.dequantize_int4_page(w, 0, 8, aa)

        u = ac.cpu().numpy().reshape(-1)
        x, z = quantize_symmetric_int4_packed(u)
        y = dequantize_symmetric_int4_packed(x, z).reshape(ac.shape)

        torch.testing.assert_close(v.cpu(), torch.from_numpy(y), atol=1e-4, rtol=1e-4)

    def test_storage_is_half_the_bytes_of_int8_for_the_same_shape(self):

        ah, ag, ad = 16, 4, 8
        af = torch.zeros(ah, ag, ad, dtype=torch.int8, device="cuda").numel()
        ae = torch.zeros(ah, ag, ad // 2, dtype=torch.uint8, device="cuda").numel()
        assert ae == af // 2


    def test_offset_write_does_not_disturb_earlier_tokens(self):
        al = torch.zeros(16, 4, 4, dtype=torch.uint8, device="cuda")
        ak = _random_tokens(8, seed=4)
        _C.quantize_int4_new_scale(ak, al, 0)
        aj = al.clone()

        am = _random_tokens(8, seed=5)


        _C.quantize_int4_new_scale(am, al, 8)

        torch.testing.assert_close(al[:8], aj[:8])
        assert not torch.equal(al[8:], aj[8:])
