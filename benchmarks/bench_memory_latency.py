import argparse
import time

import torch

from metagross.generate import generate_baseline_hf, generate_metagross, generate_metagross_fused, load_gpt2

PAGE_SIZE = 16


def bench_one(model, tokenizer, prompt, max_new_tokens):
    d = model.config
    q = len(tokenizer(prompt).input_ids) + max_new_tokens
    p = {"seq_len": q}

    for j, fn in [
        ("metagross_naive", lambda: generate_metagross(model, tokenizer, prompt, max_new_tokens=max_new_tokens, page_size=PAGE_SIZE)),
        ("metagross_fused", lambda: generate_metagross_fused(model, tokenizer, prompt, max_new_tokens=max_new_tokens, page_size=PAGE_SIZE)),
    ]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, h, _, c = fn()
        torch.cuda.synchronize()
        p[f"{j}_latency_s"] = time.perf_counter() - t0
        p[f"{j}_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        p[f"{j}_cache_mb"] = c.memory_bytes() / (1024 ** 2)
        p[f"{j}_generated"] = h

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, i, _ = generate_baseline_hf(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    p["baseline_latency_s"] = time.perf_counter() - t0
    p["baseline_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)


    p["baseline_cache_mb"] = (
        2 * d.n_layer * q * d.n_head * (d.n_embd // d.n_head) * 2
    ) / (1024 ** 2)

    p["naive_tok_match"] = sum(a == b for a, b in zip(p["metagross_naive_generated"], i))
    p["fused_tok_match"] = sum(a == b for a, b in zip(p["metagross_fused_generated"], i))
    return p


def make_plots(rows, out_path="benchmarks/results.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = [r["seq_len"] for r in rows]
    fig, v = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = v[0]
    ax.plot(z, [r["baseline_latency_s"] * 1000 for r in rows], marker="o", label="HF baseline (FP16)")
    ax.plot(z, [r["metagross_naive_latency_s"] * 1000 for r in rows], marker="o", label="Metagross (Phase 1, naive)")
    ax.plot(z, [r["metagross_fused_latency_s"] * 1000 for r in rows], marker="o", label="Metagross (Phase 2, fused)")
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Generation latency")
    ax.legend()

    ax = v[1]
    ax.plot(z, [r["baseline_cache_mb"] for r in rows], marker="o", label="HF baseline (FP16 cache)")
    ax.plot(z, [r["metagross_naive_cache_mb"] for r in rows], marker="o", label="Metagross (INT8 cache)")
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("cache memory (MB)")
    ax.set_title("KV-cache memory (Phase 1 and Phase 2 share the same storage format)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plots to {out_path}")


def main():
    ae = argparse.ArgumentParser(description=__doc__)
    ae.add_argument("--new-tokens", type=int, nargs="+", default=[16, 64, 128])
    ae.add_argument("--prompt", default="The history of artificial intelligence began with")
    ae.add_argument("--no-plots", action="store_true", help="skip matplotlib (e.g. if it isn't installed)")
    aa = ae.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU visible to this process. This benchmark needs one -- "
            "run it on the machine you built the extension on."
        )

    print("Loading GPT-2 (124M), FP16...")
    ad, ag = load_gpt2(device="cuda")

    print("Warming up (CUDA context init / first-launch overhead shouldn't pollute real measurements)...")
    bench_one(ad, ag, aa.prompt, max_new_tokens=4)

    af = []
    for n in aa.new_tokens:
        print(f"Benchmarking max_new_tokens={n}...")
        af.append(bench_one(ad, ag, aa.prompt, max_new_tokens=n))

    ac = (
        f"{'seq_len':>8} | {'baseline ms':>11} | {'naive ms':>9} | {'fused ms':>9} | "
        f"{'cache MB (hf/fkv)':>18} | {'tok match (naive/fused)':>24}"
    )
    print("\n" + ac)
    print("-" * len(ac))
    for r in af:
        print(
            f"{r['seq_len']:>8} | {r['baseline_latency_s'] * 1000:>11.1f} | "
            f"{r['metagross_naive_latency_s'] * 1000:>9.1f} | {r['metagross_fused_latency_s'] * 1000:>9.1f} | "
            f"{r['baseline_cache_mb']:>8.2f} / {r['metagross_naive_cache_mb']:>6.2f} | "
            f"{r['naive_tok_match']:>10} / {r['fused_tok_match']:<10}"
        )

    ab = sum(r["baseline_cache_mb"] / r["metagross_naive_cache_mb"] for r in af) / len(af)
    print(f"\nAverage cache-only memory ratio (baseline / metagross): {ab:.2f}x")
    print("(Expect ~2x: INT8 is 1 byte/value vs FP16's 2 bytes/value -- see module docstring for why")
    print(" this benchmark measures the quantization win specifically, not a paging win.)")

    if not aa.no_plots:
        try:
            make_plots(af)
        except ImportError:
            print("\n(matplotlib not installed -- skipping plots. `pip install matplotlib` to get them,")
            print(" or pass --no-plots to silence this message.)")


if __name__ == "__main__":
    main()
