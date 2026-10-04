import numpy as np
import pytest

from lvm_new.data import as_chunks


def test_a_generator_is_refused():
    chunks = (np.zeros((3, 2)) for _ in range(2))
    with pytest.raises(TypeError, match="one-shot"):
        as_chunks(chunks, chunk_size=10)


def test_a_1d_array_is_refused():
    with pytest.raises(ValueError, match="2-D array"):
        as_chunks(np.zeros(5), chunk_size=10)


ROWS = np.arange(46, dtype=np.float32).reshape(23, 2)


def test_an_array_is_read_in_slices_of_chunk_size_on_every_pass():
    chunks = as_chunks(ROWS, chunk_size=10)
    for _ in range(2):                                    # a second pass gives the same
        passed = list(chunks)
        assert [len(c) for c in passed] == [10, 10, 3]
        np.testing.assert_array_equal(np.concatenate(passed), ROWS)


def test_a_reiterable_passes_its_pieces_on_every_pass():
    # (features, labels) tuples, as a DataLoader yields: the features, in the loader's own chunk sizes.
    pieces = [(ROWS[a:b], np.zeros(b - a)) for a, b in [(0, 4), (4, 13), (13, 14), (14, 23)]]
    chunks = as_chunks(pieces, chunk_size=10)
    for _ in range(2):
        passed = list(chunks)
        assert [len(c) for c in passed] == [4, 9, 1, 9]
        np.testing.assert_array_equal(np.concatenate(passed), ROWS)


def test_a_list_of_points_is_refused():
    with pytest.raises(TypeError, match="Expected arrays"):
        list(as_chunks([[1.0, 2.0], [3.0, 4.0]], chunk_size=10))
