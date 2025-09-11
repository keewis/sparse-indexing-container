import numpy as np
from sparse_indexing_container.sparse_indexing_container import COO


def is_scalar(x):
    return x.ndim == 0 and not hasattr(x, "fill_value")


def compare_fill_value(a, b):
    return a == b or (np.isnan(a) and np.isnan(b))


def array_equal(a, b):
    if is_scalar(a) ^ is_scalar(b):
        if is_scalar(a):
            return b.nsv == 0 and compare_fill_value(a, b.fill_value)
        else:
            return a.nsv == 0 and compare_fill_value(a.fill_value, b)
    elif is_scalar(a) and is_scalar(b):
        return a.data == b.data

    if type(a) is not type(b):
        return False

    # unlike array-like objects, equality checks all values
    return a == b


__all__ = ["COO"]
