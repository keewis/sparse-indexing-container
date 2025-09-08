import pytest
import numpy as np

from sparse_indexing_container import COO


def test_init():
    data = np.array([True, True, True])
    coords = [np.array([1, 3, 5], dtype="uint64"), np.array([1, 4, 7], dtype="uint64")]
    shape = (7, 10)
    fill_value = False

    actual = COO(data=data, coords=coords, shape=shape, fill_value=fill_value)

    np.testing.assert_equal(actual.data, data)
    c = actual.coords
    np.testing.assert_equal(c[0], coords[0])
    np.testing.assert_equal(c[1], coords[1])

    assert actual.shape == shape
    assert actual.fill_value == fill_value
    assert actual.nsv == data.size


@pytest.mark.parametrize("shape", [(4, 4), (10, 5), (10, 15)])
@pytest.mark.parametrize(
    ["dtype", "fill_value"], (("int16", -1), ("float32", float("nan")), ("bool", False))
)
@pytest.mark.parametrize("nsv", [3, 6])
def test_repr(shape, dtype, fill_value, nsv):
    coords_ = {
        3: (np.array([0, 2, 3], dtype="uint64"), np.array([1, 3, 2], dtype="uint64")),
        6: (
            np.array([0, 1, 1, 2, 3, 3], dtype="uint64"),
            np.array([0, 0, 2, 2, 2, 3], dtype="uint64"),
        ),
    }

    coords = coords_[nsv]

    data_ = {
        ("int16", 3): np.array([12, 20, 4], dtype="int16"),
        ("int16", 6): np.array([12, 5, 120, -4, 9, 30], dtype="int16"),
        ("float32", 3): np.linspace(0, 1, 3, dtype="float32"),
        ("float32", 6): np.linspace(0, 1, 6, dtype="float32"),
        ("bool", 3): np.full(shape=3, fill_value=True),
        ("bool", 6): np.full(shape=6, fill_value=True),
    }
    data = data_[(dtype, nsv)]

    coo = COO(shape=shape, fill_value=fill_value, data=data, coords=coords)

    actual = repr(coo)

    assert actual.startswith("<COO") and actual.endswith(">")
    assert f"shape={shape}" in actual
    assert f"dtype='{dtype}'" in actual
    assert f"nsv={nsv}" in actual
    assert f"fill_value={fill_value}" in actual
