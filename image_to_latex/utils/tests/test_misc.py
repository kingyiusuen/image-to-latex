import torch

from image_to_latex.utils.misc import compute_time_elapsed, find_first, find_max_length


class TestFindMaxLength:
    def test_same_length(self):
        lst = [[1], [1], [1]]
        assert find_max_length(lst) == 1

    def test_different_lengths(self):
        lst = [[], [1], [1, 2]]
        assert find_max_length(lst) == 2


class TestComputeTimeElapsed:
    def test_zero_in_min(self):
        start_time = 30.
        end_time = 35.
        assert compute_time_elapsed(start_time, end_time) == (0, 5)

    def test_zero_in_sec(self):
        start_time = 60.
        end_time = 120.
        assert compute_time_elapsed(start_time, end_time) == (1, 0)

    def test_non_zero_min_and_sec(self):
        start_time = 0.
        end_time = 125.
        assert compute_time_elapsed(start_time, end_time) == (2, 5)


class TestFindFirst:
    def test_find_first(self):
        actual = find_first(torch.Tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
        expected = torch.LongTensor([2, 1, 3])
        assert torch.equal(actual, expected)
