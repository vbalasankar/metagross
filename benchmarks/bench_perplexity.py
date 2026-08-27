import argparse

import torch

from metagross.generate import load_gpt2
from metagross.perplexity import compute_perplexity_baseline, compute_perplexity_metagross, load_wikitext2_sample


def main():
    e = argparse.ArgumentParser(description=__doc__)
    e.add_argument("--num-chars", type=int, default=1500, help="approx. size of the WikiText-2 sample")
    e.add_argument("--page-size", type=int, default=16)
    b = e.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU visible to this process. This benchmark needs one -- "
            "run it on the machine you built the extension on."
        )

    print("Loading GPT-2 (124M), FP16...")
    c, k = load_gpt2(device="cuda")

    print(f"Loading a ~{b.num_chars}-char WikiText-2 (test split) sample...")
    j = load_wikitext2_sample(num_chars=b.num_chars)
    d = len(k(j).input_ids)
    print(f"Sample: {d} tokens.\n")

    print("Computing baseline (HF FP16, single forward pass) perplexity...")
    g = compute_perplexity_baseline(c, k, j, max_length=d)

    print("Computing Metagross Phase 1 (naive) perplexity -- one decode step per token, this is the slow part...")
    i = compute_perplexity_metagross(
        c, k, j, page_size=b.page_size, fused=False, max_length=d
    )

    print("Computing Metagross Phase 2 (fused kernel) perplexity...")
    h = compute_perplexity_metagross(
        c, k, j, page_size=b.page_size, fused=True, max_length=d
    )

    print("\n" + "-" * 50)
    print(f"{'HF baseline (FP16)':<30} {g:>10.3f}")
    print(f"{'Metagross Phase 1 (naive)':<30} {i:>10.3f}  (+{i - g:+.3f})")
    print(f"{'Metagross Phase 2 (fused)':<30} {h:>10.3f}  (+{h - g:+.3f})")
    print("-" * 50)
    print(f"Phase 1 vs Phase 2 agreement: {abs(i - h):.4f} perplexity points apart")
    print("(should be small -- both share the same quantization scheme; a large gap here points at")
    print(" a bug in one attention integration path specifically, not at quantization in general.)")


if __name__ == "__main__":
    main()
