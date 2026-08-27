import pytest

from metagross.allocator import BlockAllocator


def test_allocate_returns_distinct_pages():
    a = BlockAllocator(num_pages=4)
    b = {a.allocate() for _ in range(4)}
    assert b == {0, 1, 2, 3}


def test_allocate_raises_when_exhausted():
    c = BlockAllocator(num_pages=2)
    c.allocate()
    c.allocate()
    with pytest.raises(RuntimeError):
        c.allocate()


def test_free_makes_page_available_again():
    d = BlockAllocator(num_pages=1)
    e = d.allocate()
    with pytest.raises(RuntimeError):
        d.allocate()
    d.free(e)
    assert d.allocate() == e


def test_double_free_raises():
    f = BlockAllocator(num_pages=2)
    g = f.allocate()
    f.free(g)
    with pytest.raises(RuntimeError):
        f.free(g)


def test_free_out_of_range_raises():
    h = BlockAllocator(num_pages=2)
    with pytest.raises(ValueError):
        h.free(5)
    with pytest.raises(ValueError):
        h.free(-1)


def test_num_free_and_num_used_track_correctly():
    i = BlockAllocator(num_pages=3)
    assert (i.num_free, i.num_used) == (3, 0)
    p = i.allocate()
    assert (i.num_free, i.num_used) == (2, 1)
    i.free(p)
    assert (i.num_free, i.num_used) == (3, 0)


def test_zero_or_negative_num_pages_rejected():
    with pytest.raises(ValueError):
        BlockAllocator(num_pages=0)
    with pytest.raises(ValueError):
        BlockAllocator(num_pages=-1)
