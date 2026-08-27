class BlockAllocator:


    def __init__(self, num_pages):
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        self.num_pages = num_pages
        self._free = list(range(num_pages))

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError(
                f"BlockAllocator out of pages (all {self.num_pages} in use). "
                "Increase max_pages, or (future work) add eviction/preemption."
            )
        return self._free.pop()

    def free(self, page_idx) -> None:
        if not (0 <= page_idx < self.num_pages):
            raise ValueError(f"page_idx {page_idx} out of range [0, {self.num_pages})")


        if page_idx in self._free:
            raise RuntimeError(f"page_idx {page_idx} is already free (double free)")
        self._free.append(page_idx)

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_pages - len(self._free)
