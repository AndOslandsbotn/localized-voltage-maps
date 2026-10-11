import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common.datasets import FileChunks, mnist8m_labels, stream
from experiments.common.guard import watch_memory
from experiments.common.record import environment, model_record
from lvm import LocalizedVoltageMaps
from lvm.cells import assign_cells
from lvm.config import load_config
from lvm.region import fit_region, fit_region_chart, save_region

HERE = Path(__file__).resolve().parent
DATASET, N_FEATURES = "mnist8m", 784
SIZES = (5_000, 10_000, 20_000, 50_000, 100_000, 150_000, 200_000, 300_000)
SEEDS = tuple(range(10))
REFINE = {"cells": {"refine": {"strategy": "lloyd", "passes": 10, "tolerance": 0.05}, "masses": {"rel_error": None},
                    "kmeans": {"lloyd": {"iterations": 3}}}}
QUERIES, K = 5_000, 10
MIN_FOR_DIMENSION = 100


def window(n: int, seed: int) -> int:
    """First row of the n points: always the start of the shuffled file, so the data sets are nested (5k in 10k in
    ...) and the same for every seed and method; the seeds vary only the methods."""
    return 0


def read(n: int, seed: int) -> np.ndarray:
    """Seed's n points read from disk into one array (the data is never held twice)."""
    X, offset = np.empty((n, N_FEATURES), dtype=np.float32), 0
    for chunk in stream(DATASET, start=window(n, seed), n=n):
        X[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
    return X


def points(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seed's n points and their labels, as every method reads them."""
    start = window(n, seed)
    labels = mnist8m_labels(start + n)[start:]
    return read(n, seed), labels


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    warm_up = stream(DATASET, n=2000)
    LocalizedVoltageMaps(random_state=0, config=REFINE).fit(warm_up).transform(warm_up)
    for n in SIZES:
        for seed in SEEDS:
            path = HERE / "embeddings" / f"lvm_n{n}_seed{seed}.npz"
            if not path.exists():
                run(n, seed, path)


def run(n: int, seed: int, path: Path) -> None:
    data = stream(DATASET, start=window(n, seed), n=n)
    seconds = {}

    start = time.perf_counter()
    level1 = LocalizedVoltageMaps(local_chart="none", random_state=seed, config=REFINE).fit(data)
    seconds["level1_fit"] = time.perf_counter() - start
    start = time.perf_counter()
    Z1 = level1.transform(data)
    seconds["level1_place"] = time.perf_counter() - start
    root = level1.root_
    start = time.perf_counter()
    level1.fit_charts(data)
    seconds["level1_chart_fit"] = time.perf_counter() - start
    start = time.perf_counter()
    Z1_chart = level1.transform(data)
    seconds["level1_chart_place"] = time.perf_counter() - start
    models = HERE / "models" / f"n{n}_seed{seed}"
    level1.save(models / "level1")
    (models / "level2").mkdir(exist_ok=True)

    cell_region = root.maps.argmax(dim=0)
    buckets = HERE / "buckets"
    start = time.perf_counter()
    region_of_point = bucket(data, root.centroids, cell_region, buckets)
    seconds["bucketing"] = time.perf_counter() - start

    Z2, Z2_chart = Z1.copy(), Z1_chart.copy()
    region_seconds, passes = {}, {}
    for r in np.unique(region_of_point):
        members = np.flatnonzero(region_of_point == r)
        if len(members) < 3:
            continue
        chunks = FileChunks(buckets / f"region_{r}.u8", 0, len(members), N_FEATURES, 10_000)
        dimension = ({} if len(members) >= MIN_FOR_DIMENSION
                     else {"dimension": {"strategy": "fixed", "fixed": {"d": root.dimension}}})
        config = load_config(overrides={**REFINE, **dimension, "embedding": {"local_chart": "none"}})
        times = {}
        start = time.perf_counter()
        region = fit_region(chunks, config, device="cuda", region_id=(int(r),), seed=seed,
                            parents=root.centroids[cell_region == r])
        times["fit"] = time.perf_counter() - start
        start = time.perf_counter()
        Z2[members] = place(region, chunks, config)
        times["place"] = time.perf_counter() - start
        start = time.perf_counter()
        region = fit_region_chart(region, chunks, config=config, device="cuda")
        times["chart_fit"] = time.perf_counter() - start
        start = time.perf_counter()
        Z2_chart[members] = place(region, chunks, config)
        times["chart_place"] = time.perf_counter() - start
        region_seconds[int(r)], passes[int(r)] = times, region.refine_passes
        save_region(region, models / "level2" / f"region_{r}.npz")
    shutil.rmtree(buckets)

    np.savez(path, start=window(n, seed), level1=Z1, level1_chart=Z1_chart, level2=Z2, level2_chart=Z2_chart,
             region_of_point=region_of_point, neighbours_in_region=neighbours_in_region(data, region_of_point, seed),
             seconds=json.dumps(seconds), region_seconds=json.dumps(region_seconds), passes=json.dumps(passes),
             level1_model=json.dumps(model_record(level1)), environment=json.dumps(environment("cuda")))


def place(region, chunks, config) -> np.ndarray:
    return np.concatenate([region.transform_chunk(torch.as_tensor(chunk, device="cuda"), config=config).cpu().numpy()
                           for chunk in chunks])


def bucket(data, centroids: torch.Tensor, cell_region: torch.Tensor, folder: Path) -> np.ndarray:
    """One pass over the data: each point's region (the landmark basin of its nearest level-1 cell), with every
    point written, in order, to its region's file as uint8 rows."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    files, regions = {}, []
    for chunk in data:
        region = cell_region[assign_cells(torch.as_tensor(chunk, device=centroids.device), centroids)].cpu().numpy()
        raw = np.rint(chunk * 255.0).astype(np.uint8)
        for r in np.unique(region):
            if r not in files:
                files[r] = open(folder / f"region_{r}.u8", "wb")
            raw[region == r].tofile(files[r])
        regions.append(region)
    for f in files.values():
        f.close()
    return np.concatenate(regions)


def neighbours_in_region(data, region_of_point: np.ndarray, seed: int) -> float:
    """Share of the true K nearest neighbours (among all n points) of QUERIES random points that lie in the query's
    own region."""
    n = len(region_of_point)
    rows = np.sort(np.random.default_rng(seed).choice(n, min(QUERIES, n), replace=False))
    queries = torch.as_tensor(np.concatenate(list(data))[rows] if n <= QUERIES else _rows(data, rows), device="cuda")
    best_d = torch.full((len(rows), K + 1), float("inf"), device="cuda")
    best_i = torch.zeros((len(rows), K + 1), dtype=torch.long, device="cuda")
    offset = 0
    for chunk in data:
        d = torch.cdist(queries, torch.as_tensor(chunk, device="cuda"))
        index = torch.arange(offset, offset + len(chunk), device="cuda").expand(len(rows), -1)
        merged_d, order = torch.topk(torch.cat([best_d, d], dim=1), K + 1, dim=1, largest=False)
        best_i = torch.gather(torch.cat([best_i, index], dim=1), 1, order)
        best_d = merged_d
        offset += len(chunk)
    neighbours = best_i[:, 1:].cpu().numpy()
    return float(np.mean(region_of_point[neighbours] == region_of_point[rows][:, None]))


def _rows(data, rows: np.ndarray) -> np.ndarray:
    kept, offset = [], 0
    for chunk in data:
        inside = rows[(rows >= offset) & (rows < offset + len(chunk))]
        kept.append(chunk[inside - offset])
        offset += len(chunk)
    return np.concatenate(kept)


if __name__ == "__main__":
    main()
