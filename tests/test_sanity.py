import pytest
import torch

import metagross

pytest.importorskip(
    "metagross._C",
    reason="metagross._C extension not built yet -- run "
    "`pip install -e . --no-build-isolation` first.",
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")
def test_sanity_add_matches_torch():
    torch.manual_seed(0)
    a = torch.randn(10_000, device="cuda", dtype=torch.float32)
    b = torch.randn(10_000, device="cuda", dtype=torch.float32)

    out = metagross.sanity_add(a, b)

    torch.testing.assert_close(out, a + b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")
def test_sanity_add_shape_mismatch_raises():
    a = torch.randn(10, device="cuda")
    b = torch.randn(20, device="cuda")
    with pytest.raises(RuntimeError):
        metagross.sanity_add(a, b)


if __name__ == "__main__":

    test_sanity_add_matches_torch()
    test_sanity_add_shape_mismatch_raises()
    print("Phase 0 sanity check PASSED: extension built, loaded, and executed correctly.")
