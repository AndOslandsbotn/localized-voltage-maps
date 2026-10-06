import numpy as np
import pytest

from lvm_old.cells import assign_cells
from lvm_old.config import load_config
from lvm_old.regions import Region, route, sample_regions
from lvm_old.stream import array_source, iter_array_chunks, iter_csv_chunks, make_source


def test_csv_chunks_match_file(tmp_path):
    X = np.random.default_rng(0).random((23, 3))
    path = tmp_path / "data.csv"
    np.savetxt(path, X, delimiter=",", header="a,b,c", comments="")
    chunks = list(iter_csv_chunks(path, 5, skip_header=1))
    assert [c.shape[0] for c in chunks] == [5, 5, 5, 5, 3]
    np.testing.assert_allclose(np.vstack(chunks), X)


def test_make_source_reads_csv_from_config(tmp_path):
    X = np.arange(12.0).reshape(6, 2)
    path = tmp_path / "data.csv"
    np.savetxt(path, X, delimiter=";")
    cfg = load_config(overrides={"data": {"path": str(path), "chunk_size": 4, "csv": {"delimiter": ";"}}})
    source = make_source(cfg.data)
    # A source can be scanned more than once (one scan per level).
    for _ in range(2):
        np.testing.assert_allclose(np.vstack(list(source())), X)


def test_array_chunks_cover_array():
    X = np.arange(20.0).reshape(10, 2)
    np.testing.assert_array_equal(np.vstack(list(iter_array_chunks(X, 3))), X)


def test_assign_cells_matches_brute_force():
    rng = np.random.default_rng(1)
    X, C = rng.random((200, 4)), rng.random((15, 4))
    expected = np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), axis=1)
    np.testing.assert_array_equal(assign_cells(X, C), expected)


def _two_level_tree():
    # Level 0: 1-D line with 4 cells centred at 0, 1, 2, 3.
    root = Region(centroids=np.array([[0.0], [1.0], [2.0], [3.0]]))
    left = root.add_child([0, 1, 2])   # overlaps `right` on cell 2
    right = root.add_child([2, 3])
    # Level 1: split `left` into two cells, each its own child.
    left.centroids = np.array([[0.0], [2.0]])
    left.add_child([0])
    left.add_child([1])
    return root


def test_route_overlapping_children_and_depth():
    root = _two_level_tree()
    chunk = np.array([[0.1], [0.9], [2.1], [3.2]])
    routed = {leaf.id: rows.tolist() for leaf, rows in route(chunk, root)}
    assert routed == {
        (0, 0): [0, 1],   # left, nearest of {0, 2} is 0
        (0, 1): [2],      # left, nearest of {0, 2} is 2
        (1,): [2, 3],     # right: cells 2 and 3; point 2.1 is in both left and right
    }


def test_route_drops_points_in_no_child():
    root = Region(centroids=np.array([[0.0], [1.0]]))
    root.add_child([0])
    routed = {leaf.id: rows.tolist() for leaf, rows in route(np.array([[0.0], [1.0]]), root)}
    assert routed == {(0,): [0]}


def test_add_child_requires_cells():
    with pytest.raises(ValueError, match="no cells yet"):
        Region().add_child([0])


def test_shuffled_sampling_stops_early():
    X = np.arange(100.0)[:, None]
    chunks_read = 0

    def source():
        nonlocal chunks_read
        for chunk in iter_array_chunks(X, 10):
            chunks_read += 1
            yield chunk

    samples = sample_regions(source, Region(), sample_size=25, shuffled=True)
    np.testing.assert_array_equal(samples[()].points[:, 0], np.arange(25.0))
    assert chunks_read == 3


def test_sampling_per_region_counts_and_sizes():
    root = _two_level_tree()
    X = np.random.default_rng(2).uniform(-0.4, 3.4, size=(5000, 1))
    samples = sample_regions(array_source(X, 700), root, sample_size=50, shuffled=False, seed=0)
    expected = {leaf.id: rows.size for leaf, rows in route(X, root)}
    assert {rid: s.n_seen for rid, s in samples.items()} == expected
    for rid, s in samples.items():
        assert s.points.shape == (50, 1)
        # Every sampled point really belongs to its region (and maybe to an overlapping one).
        member_rows = {leaf.id: rows for leaf, rows in route(s.points, root)}
        assert member_rows[rid].size == s.points.shape[0]


def test_reservoir_sample_is_uniform():
    # Each of the 1000 points should be kept with probability 100/1000 = 0.1,
    # regardless of where it sits in the stream (including across chunk borders).
    X = np.arange(1000.0)[:, None]
    hits = np.zeros(1000)
    runs = 400
    for seed in range(runs):
        s = sample_regions(array_source(X, 64), Region(), sample_size=100, shuffled=False, seed=seed)[()]
        assert np.unique(s.points).size == 100
        hits[s.points[:, 0].astype(int)] += 1
    freq = hits / runs
    # Averages over 100-point blocks: std about sqrt(0.1*0.9/(400*100)) = 0.0015.
    np.testing.assert_allclose(freq.reshape(10, 100).mean(axis=1), 0.1, atol=0.01)
