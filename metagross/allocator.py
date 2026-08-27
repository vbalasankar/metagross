class BlockAllocator:


    def __init__(b, a: int):
        if a <= 0:
            raise ValueError("num_pages must be positive")
        b.num_pages = a
        b._free = list(range(a))

    def allocate(c) -> int:
        if not c._free:
            raise RuntimeError(
                f"BlockAllocator out of pages (all {c.num_pages} in use). "
                "Increase max_pages, or (future work) add eviction/preemption."
            )
        return c._free.pop()

    def free(e, d: int) -> None:
        if not (0 <= d < e.num_pages):
            raise ValueError(f"page_idx {d} out of range [0, {e.num_pages})")


        if d in e._free:
            raise RuntimeError(f"page_idx {d} is already free (double free)")
        e._free.append(d)

    @property
    def num_free(g) -> int:
        return len(g._free)

    @property
    def num_used(h) -> int:
        return h.num_pages - len(h._free)
