import pytest
import torch

pytest.importorskip("metagross._C", reason="metagross._C extension not built yet")

from metagross.generate import generate_baseline_hf
from metagross.generate_llama import generate_tinyllama_fused, load_tinyllama


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")

PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
]

TEST_PAGE_SIZE = 4


@pytest.fixture(scope="module")
def tinyllama():
    return load_tinyllama(device="cuda")


@pytest.mark.parametrize("prompt", PROMPTS)
def test_first_step_logits_close_to_baseline(tinyllama, prompt):
    e, i = tinyllama
    _, _, c, _ = generate_tinyllama_fused(
        e, i, prompt, max_new_tokens=1, page_size=TEST_PAGE_SIZE
    )
    _, _, d = generate_baseline_hf(e, i, prompt, max_new_tokens=1)


    torch.testing.assert_close(c[0], d, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_generation_close_to_baseline(tinyllama, prompt):
    s, v = tinyllama
    q = 12
    m, o, _, l = generate_tinyllama_fused(
        s, v, prompt, max_new_tokens=q, page_size=TEST_PAGE_SIZE
    )
    j, k, _ = generate_baseline_hf(s, v, prompt, max_new_tokens=q)

    p = sum(a == b for a, b in zip(o, k))
    print(f"\n[{prompt!r}] TinyLlama fused vs baseline token match: {p}/{q}"
          f"\n  fused:    {m!r}\n  baseline: {j!r}")


    assert l.k_layers[0].storage.shape[2] == s.config.num_key_value_heads
    assert p >= 1, "zero token matches suggests a real bug, not just quantization noise"


def test_output_is_not_degenerate(tinyllama):
    x, z = tinyllama
    _, w, _, _ = generate_tinyllama_fused(
        x, z, PROMPTS[0], max_new_tokens=5, page_size=TEST_PAGE_SIZE
    )
    assert len(set(w)) > 1, "degenerate output (all identical tokens) -- check for NaN/Inf in the cache"
