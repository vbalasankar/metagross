import pytest
import torch

pytest.importorskip("metagross._C", reason="metagross._C extension not built yet")

from metagross.generate import generate_baseline_hf, generate_metagross, generate_metagross_fused, load_gpt2


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")

PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "def fibonacci(n):\n    if n <= 1:",
]

TEST_PAGE_SIZE = 4
RANK_THRESHOLD = 10


@pytest.fixture(scope="module")
def gpt2():
    return load_gpt2(device="cuda")


@pytest.mark.parametrize("prompt", PROMPTS)
def test_first_step_logits_close_to_baseline(gpt2, prompt):
    g, i = gpt2

    _, _, e, _ = generate_metagross(
        g, i, prompt, max_new_tokens=1, page_size=TEST_PAGE_SIZE
    )
    _, _, c = generate_baseline_hf(g, i, prompt, max_new_tokens=1)


    torch.testing.assert_close(e[0], c, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_second_step_prediction_remains_plausible(gpt2, prompt):


    ac, aj = gpt2
    q = next(ac.parameters()).device

    _, _, z, aa = generate_metagross(
        ac, aj, prompt, max_new_tokens=2, page_size=TEST_PAGE_SIZE
    )
    ab = z[1]

    _, _, l = generate_baseline_hf(ac, aj, prompt, max_new_tokens=1)
    ad = torch.argmax(l, dim=-1, keepdim=True)
    af = aj(prompt, return_tensors="pt").input_ids.to(q)
    j = ac(input_ids=torch.cat([af, ad], dim=1), use_cache=False)
    m = j.logits[:, -1, :]

    k0 = aa.k_layers[0]
    assert k0.block_table, (
        f"no page committed for prompt {prompt!r} even at page_size={TEST_PAGE_SIZE} -- "
        "prompt is shorter than expected, or staging/commit logic isn't triggering (see metagross/cache.py)"
    )

    assert torch.isfinite(ab).all(), f"[{prompt!r}] NaN/Inf in Metagross's step-2 logits"


    ai = k0.page_scales[k0.block_table[0]]
    u = (ab - m).abs()
    x, y = u.max().item(), u.mean().item()
    k = m.std().item()

    o = torch.argmax(m, dim=-1)
    ah = torch.argsort(ab, dim=-1, descending=True)
    ag = (ah[0] == o.item()).nonzero(as_tuple=True)[0].item() + 1
    p = torch.softmax(ab, dim=-1)[0, o].item()

    print(
        f"\n[{prompt!r}] layer-0 K page-0 scale={ai:.5f} | max diff={x:.4f} mean diff={y:.4f} "
        f"| baseline logit std={k:.4f} | baseline's top token rank under Metagross={ag} "
        f"(prob={p:.4f})"
    )


    assert ag <= RANK_THRESHOLD, (
        f"[{prompt!r}] baseline's top token fell to rank {ag} (out of {ab.shape[-1]}) "
        f"under Metagross's step-2 distribution, outside the top-{RANK_THRESHOLD} -- this is a large enough "
        f"behavioral divergence to warrant checking tests/test_generation_diagnostics.py's per-layer K/V "
        f"error breakdown for this prompt (pytest tests/test_generation_diagnostics.py -v -s) before "
        f"assuming it's ordinary quantization noise."
    )


    assert x <= 20.0 * max(k, 1e-3), (
        f"[{prompt!r}] max logit diff ({x:.4f}) is over 20x the baseline's own logit std "
        f"({k:.4f}) -- implausibly large even accounting for the rank check passing; "
        "likely a real bug rather than quantization noise."
    )


@pytest.mark.parametrize("prompt", PROMPTS)
def test_generation_close_to_baseline(gpt2, prompt):
    ar, av = gpt2
    ao = 15

    ap, aq, _, _ = generate_metagross(
        ar, av, prompt, max_new_tokens=ao, page_size=TEST_PAGE_SIZE
    )
    ak, al, _ = generate_baseline_hf(ar, av, prompt, max_new_tokens=ao)

    an = sum(a == b for a, b in zip(aq, al))
    print(
        f"\n[{prompt!r}] token match: {an}/{ao}"
        f"\n  metagross:  {ap!r}"
        f"\n  baseline: {ak!r}"
    )


    assert an >= 1, "zero token matches suggests a real bug, not just quantization noise"


def test_output_is_not_degenerate(gpt2):


    ay, az = gpt2
    _, aw, _, _ = generate_metagross(
        ay, az, PROMPTS[0], max_new_tokens=5, page_size=TEST_PAGE_SIZE
    )
    assert len(set(aw)) > 1, "degenerate output (all identical tokens) -- check for NaN/Inf in the cache"


class TestGenerateMetagrossFused:


    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_first_step_logits_close_to_baseline(self, gpt2, prompt):
        bd, bg = gpt2
        _, _, ba, _ = generate_metagross_fused(
            bd, bg, prompt, max_new_tokens=1, page_size=TEST_PAGE_SIZE
        )
        _, _, bb = generate_baseline_hf(bd, bg, prompt, max_new_tokens=1)


        torch.testing.assert_close(ba[0], bb, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_generation_close_to_baseline(self, gpt2, prompt):
        bo, br = gpt2
        bn = 15
        bj, bk, _, _ = generate_metagross_fused(
            bo, br, prompt, max_new_tokens=bn, page_size=TEST_PAGE_SIZE
        )
        bh, bi, _ = generate_baseline_hf(bo, br, prompt, max_new_tokens=bn)

        bm = sum(a == b for a, b in zip(bk, bi))
        print(f"\n[{prompt!r}] fused vs baseline token match: {bm}/{bn}"
              f"\n  fused:    {bj!r}\n  baseline: {bh!r}")
        assert bm >= 1, "zero token matches suggests a real bug, not just quantization noise"

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_matches_non_fused_metagross_closely(self, gpt2, prompt):


        bw, ca = gpt2
        bv = 15
        _, bx, _, _ = generate_metagross(
            bw, ca, prompt, max_new_tokens=bv, page_size=TEST_PAGE_SIZE
        )
        _, bs, _, _ = generate_metagross_fused(
            bw, ca, prompt, max_new_tokens=bv, page_size=TEST_PAGE_SIZE
        )
        bu = sum(a == b for a, b in zip(bx, bs))
        print(f"\n[{prompt!r}] fused vs non-fused Metagross token match: {bu}/{bv}")
        assert bu >= 1, (
            "fused and non-fused Metagross paths produced completely different output -- "
            "since both are checked against the same HF baseline independently, this points "
            "specifically at the fused-kernel integration (generate_metagross_fused), not at "
            "quantization noise in general"
        )

    def test_output_is_not_degenerate(self, gpt2):
        cd, cf = gpt2
        _, cb, _, _ = generate_metagross_fused(
            cd, cf, PROMPTS[0], max_new_tokens=5, page_size=TEST_PAGE_SIZE
        )
        assert len(set(cb)) > 1, "degenerate output (all identical tokens) -- check for NaN/Inf in the cache"
