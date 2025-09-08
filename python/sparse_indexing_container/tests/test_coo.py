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
