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


def test_oindex():
    data = np.array([True, True, True])
    coords = [np.array([1, 3, 5], dtype="uint64"), np.array([1, 4, 7], dtype="uint64")]
    shape = (7, 10)
    fill_value = False

    coo = COO(data=data, coords=coords, shape=shape, fill_value=fill_value)

    actual = coo.oindex((slice(0, 5), slice(0, 5)))
    assert actual.shape == (5, 5)
    assert actual.nsv == 2
    assert actual.dtype == coo.dtype
    assert actual.fill_value == coo.fill_value

    np.testing.assert_equal(actual.data, data[:2])
    np.testing.assert_equal(actual.coords[0], coords[0][:2])
    np.testing.assert_equal(actual.coords[1], coords[1][:2])

    actual = coo.oindex((slice(3, 7), slice(5, 10)))
    assert actual.shape == (4, 5)
    assert actual.nsv == 1
    assert actual.dtype == coo.dtype
    assert actual.fill_value == coo.fill_value

    np.testing.assert_equal(actual.data, data[2:])
    np.testing.assert_equal(actual.coords[0], np.array([2], dtype="uint64"))
    np.testing.assert_equal(actual.coords[1], np.array([2], dtype="uint64"))


@pytest.mark.parametrize(["nsv", "other_nsv"], ((3, 3), (3, 4), (4, 3)))
@pytest.mark.parametrize("shape", ((4,), (4, 4)))
@pytest.mark.parametrize("other_shape", ((4,), (4, 4)))
@pytest.mark.parametrize("dtype", ("float64", "bool"))
@pytest.mark.parametrize("other_dtype", ("float64", "bool"))
def test_eq_params(nsv, shape, dtype, other_nsv, other_shape, other_dtype):
    coords = {
        3: [np.array([1, 3, 5], dtype="uint64"), np.array([1, 4, 7], dtype="uint64")],
        4: [
            np.array([1, 3, 4, 5], dtype="uint64"),
            np.array([1, 4, 6, 7], dtype="uint64"),
        ],
    }
    data = {
        (3, "bool"): np.array([True, True, True]),
        (4, "bool"): np.array([True, True, True, True]),
        (3, "float64"): np.array([5, 2, 0], dtype="float64"),
        (4, "float64"): np.array([6, 9, 10, 12], dtype="float64"),
    }
    fill_value = {
        "bool": False,
        "float64": np.nan,
    }

    data1 = data[(nsv, dtype)]
    data2 = data[(other_nsv, other_dtype)]

    coords1 = coords[nsv]
    coords2 = coords[other_nsv]

    shape1 = shape
    shape2 = other_shape

    fill_value1 = fill_value[dtype]
    fill_value2 = fill_value[other_dtype]

    coo1 = COO(data=data1, coords=coords1, shape=shape1, fill_value=fill_value1)
    coo2 = COO(data=data2, coords=coords2, shape=shape2, fill_value=fill_value2)

    expected = (nsv == other_nsv) and (shape == other_shape) and (dtype == other_dtype)

    assert (coo1 == coo2) == expected


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("other_index", range(3))
def test_eq_data(index, other_index):
    data = [
        np.linspace(0, 10, 5, dtype="float32"),
        np.arange(5, dtype="float32"),
        np.linspace(-1, 1, 5, dtype="float32"),
    ]

    data1 = data[index]
    data2 = data[other_index]

    coords = [np.arange(5, dtype="uint64"), np.arange(5, dtype="uint64")]
    shape = (6, 6)
    fill_value = 0

    coo1 = COO(data=data1, coords=coords, shape=shape, fill_value=fill_value)
    coo2 = COO(data=data2, coords=coords, shape=shape, fill_value=fill_value)

    expected = index == other_index

    assert (coo1 == coo2) == expected
