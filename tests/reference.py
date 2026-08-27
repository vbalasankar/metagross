import numpy as np


def quantize_symmetric_int8(values) -> tuple[np.ndarray, float]:


    a = np.abs(values).max()
    b = a / 127.0 if a > 1e-12 else 1e-8
    q = np.round(values / b)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, float(b)


def dequantize_symmetric_int8(q: np.ndarray, scale) -> np.ndarray:
    return q.astype(np.float32) * np.float32(scale)


def quantize_symmetric_int4_packed(values) -> tuple[np.ndarray, float]:


    assert values.ndim == 1 and values.shape[0] % 2 == 0
    f = np.abs(values).max()
    h = f / 7.0 if f > 1e-12 else 1e-8
    q = np.clip(np.round(values / h), -7, 7).astype(np.int8)
    g = np.zeros(len(values) // 2, dtype=np.uint8)
    for i in range(0, len(values), 2):
        low = int(q[i]) & 0x0F
        e = int(q[i + 1]) & 0x0F
        g[i // 2] = low | (e << 4)
    return g, float(h)


def dequantize_symmetric_int4_packed(packed, scale) -> np.ndarray:
    out = np.zeros(len(packed) * 2, dtype=np.float32)
    for i, k in enumerate(packed):
        n, m = int(k) & 0x0F, (int(k) >> 4) & 0x0F
        low = n - 16 if n >= 8 else n
        l = m - 16 if m >= 8 else m
        out[2 * i] = low * scale
        out[2 * i + 1] = l * scale
    return out


def max_roundtrip_error(values) -> float:

    q, s = quantize_symmetric_int8(values)
    r = dequantize_symmetric_int8(q, s)
    return float(np.abs(values.astype(np.float32) - r).max())
