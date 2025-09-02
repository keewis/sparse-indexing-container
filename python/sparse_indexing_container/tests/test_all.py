import pytest
import sparse_indexing_container


def test_sum_as_string():
    assert sparse_indexing_container.sum_as_string(1, 1) == "2"
