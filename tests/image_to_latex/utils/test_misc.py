import pytest
import torch

from image_to_latex.utils.misc import (
    compute_time_elapsed,
    find_first,
    find_max_length,
)


class TestFindMaxLength:
    def test_same_length(self):
        lst = [[1], [1], [1]]
        assert find_max_length(lst) == 1

    def test_different_lengths(self):
        lst = [[], [1], [1, 2]]
        assert find_max_length(lst) == 2


@pytest.mark.parametrize(
    "inputs, expected",
    [
        ((30, 35), (0, 5)),
        ((60, 120), (1, 0)),
        ((0, 125), (2, 5)),
        ((0, 3723), (62, 3)),
    ],
)
def test_compute_test_elapsed(inputs, expected):
    assert compute_time_elapsed(*inputs) == expected


def test_find_first():
    actual = find_first(torch.Tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
    expected = torch.LongTensor([2, 1, 3])
    assert torch.equal(actual, expected)
