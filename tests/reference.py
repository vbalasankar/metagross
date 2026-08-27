import numpy as np


def quantize_symmetric_int8(c: np.ndarray) -> tuple[np.ndarray, float]:


    a = np.abs(c).max()
    b = a / 127.0 if a > 1e-12 else 1e-8
    q = np.round(c / b)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, float(b)


def dequantize_symmetric_int8(q: np.ndarray, d: float) -> np.ndarray:
    return q.astype(np.float32) * np.float32(d)


def quantize_symmetric_int4_packed(j: np.ndarray) -> tuple[np.ndarray, float]:


    assert j.ndim == 1 and j.shape[0] % 2 == 0
    f = np.abs(j).max()
    h = f / 7.0 if f > 1e-12 else 1e-8
    q = np.clip(np.round(j / h), -7, 7).astype(np.int8)
    g = np.zeros(len(j) // 2, dtype=np.uint8)
    for i in range(0, len(j), 2):
        low = int(q[i]) & 0x0F
        e = int(q[i + 1]) & 0x0F
        g[i // 2] = low | (e << 4)
    return g, float(h)


def dequantize_symmetric_int4_packed(o: np.ndarray, p: float) -> np.ndarray:
    out = np.zeros(len(o) * 2, dtype=np.float32)
    for i, k in enumerate(o):
        n, m = int(k) & 0x0F, (int(k) >> 4) & 0x0F
        low = n - 16 if n >= 8 else n
        l = m - 16 if m >= 8 else m
        out[2 * i] = low * p
        out[2 * i + 1] = l * p
    return out


def max_roundtrip_error(t: np.ndarray) -> float:

    q, s = quantize_symmetric_int8(t)
    r = dequantize_symmetric_int8(q, s)
    return float(np.abs(t.astype(np.float32) - r).max())
