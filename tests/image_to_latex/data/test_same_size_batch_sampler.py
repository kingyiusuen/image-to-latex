import pytest
import torch

from image_to_latex.data.base_dataset import BaseDataset
from image_to_latex.data.same_size_batch_sampler import (
    SameSizeBatchSampler,
    _get_buckets,
)


class TestSameSizeBatchSampler:
    @pytest.fixture
    def dataset(self):
        data = [
            torch.rand(1, 1, 1),
            torch.rand(1, 1, 1),
            torch.rand(1, 2, 2),
            torch.rand(1, 2, 2),
            torch.rand(1, 2, 2),
            torch.rand(1, 3, 3),
            torch.rand(1, 3, 3),
            torch.rand(1, 3, 3),
        ]
        targets = torch.rand(len(data))
        return BaseDataset(data, targets)

    def test_get_buckets(self, dataset):
        expected = {
            (1, 1, 1): [0, 1],
            (1, 2, 2): [2, 3, 4],
            (1, 3, 3): [5, 6, 7],
        }
        assert _get_buckets(dataset) == expected

    @pytest.mark.parametrize(
        "batch_size, expected",
        [
            (1, 8),
            (2, 5),
            (3, 3),
            (4, 3),
        ],
    )
    def test_len_without_drop_last(self, dataset, batch_size, expected):
        sampler = SameSizeBatchSampler(dataset, batch_size, drop_last=False)
        assert len(sampler) == expected

    @pytest.mark.parametrize(
        "batch_size, expected",
        [
            (1, 8),
            (2, 3),
            (3, 2),
            (4, 0),
        ],
    )
    def test_len_with_drop_last(self, dataset, batch_size, expected):
        sampler = SameSizeBatchSampler(dataset, batch_size, drop_last=True)
        assert len(sampler) == expected

    @pytest.mark.parametrize(
        "batch_size, expected",
        [
            (1, [[0], [1], [2], [3], [4], [5], [6], [7]]),
            (2, [[0, 1], [2, 3], [4], [5, 6], [7]]),
            (3, [[0, 1], [2, 3, 4], [5, 6, 7]]),
            (4, [[0, 1], [2, 3, 4], [5, 6, 7]]),
        ],
    )
    def test_iter_without_drop_last(self, dataset, batch_size, expected):
        sampler = SameSizeBatchSampler(dataset, batch_size, drop_last=False)
        actual = [batch for batch in sampler]
        assert actual == expected

    @pytest.mark.parametrize(
        "batch_size, expected",
        [
            (1, [[0], [1], [2], [3], [4], [5], [6], [7]]),
            (2, [[0, 1], [2, 3], [5, 6]]),
            (3, [[2, 3, 4], [5, 6, 7]]),
            (4, []),
        ],
    )
    def test_iter_with_drop_last(self, dataset, batch_size, expected):
        sampler = SameSizeBatchSampler(dataset, batch_size, drop_last=True)
        actual = [batch for batch in sampler]
        assert actual == expected
