import numpy as np

from experiments.common.datasets import DATA, MNIST8M_DIR, MNIST8M_N, mnist8m_labels, stream

N_FEATURES = 784


def read(dataset: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The first n points of a shuffled on-disk dataset (kuzushiji49, mnist8m) and their labels."""
    X, offset = np.empty((n, N_FEATURES), dtype=np.float32), 0
    for chunk in stream(dataset, n=n):
        X[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
    labels = (mnist8m_labels(n) if dataset == "mnist8m"
              else np.fromfile(DATA / dataset / "y.i8", dtype=np.int8)[:n].astype(np.int64))
    return X, labels


def on_disk(dataset: str) -> np.ndarray:
    """All of a shuffled on-disk dataset as a read-only uint8 memory map (rows of N_FEATURES), for picking rows."""
    if dataset == "mnist8m":
        return np.memmap(MNIST8M_DIR / "X.u8", dtype=np.uint8, mode="r", shape=(MNIST8M_N, N_FEATURES))
    path = DATA / dataset / "X.u8"
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(path.stat().st_size // N_FEATURES, N_FEATURES))


def read_rows(dataset: str, rows: np.ndarray) -> np.ndarray:
    """(len(rows), N_FEATURES) float32 images at the given rows, read with plain file reads (sorted, contiguous runs
    read together), so nothing stays mapped in memory."""
    path = MNIST8M_DIR / "X.u8" if dataset == "mnist8m" else DATA / dataset / "X.u8"
    rows = np.sort(rows)
    out = np.empty((len(rows), N_FEATURES), dtype=np.float32)
    if len(rows) == 0:
        return out
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    with open(path, "rb") as f:
        for start, stop in zip(np.r_[0, breaks], np.r_[breaks, len(rows)]):
            f.seek(int(rows[start]) * N_FEATURES)
            block = np.fromfile(f, dtype=np.uint8, count=(stop - start) * N_FEATURES).reshape(stop - start, N_FEATURES)
            out[start:stop] = block.astype(np.float32) / 255.0
    return out
