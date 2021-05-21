import pytest

from image_to_latex.data.base_dataset import BaseDataset


def square(x):
    return x ** 2


def divide_by_two(x):
    return x // 2


class TestBaseDataset:
    @pytest.fixture
    def data(self):
        return [1, 2, 3]

    @pytest.fixture
    def targets(self):
        return [2, 4, 6]

    def test_len(self, data, targets):
        dataset = BaseDataset(data, targets, None, None)
        assert len(dataset) == 3

    def test_getitem(self, data, targets):
        dataset = BaseDataset(data, targets, None, None)
        actual = [dataset[i] for i in range(len(dataset))]
        expected = [(1, 2), (2, 4), (3, 6)]
        assert actual == expected

    def test_transform(self, data, targets):
        dataset = BaseDataset(data, targets, square, None)
        actual = [dataset[i] for i in range(len(dataset))]
        expected = [(1, 2), (4, 4), (9, 6)]
        assert actual == expected

    def test_target_transform(self, data, targets):
        dataset = BaseDataset(data, targets, None, divide_by_two)
        actual = [dataset[i] for i in range(len(dataset))]
        expected = [(1, 1), (2, 2), (3, 3)]
        assert actual == expected

    def test_transform_and_target_transform(self, data, targets):
        dataset = BaseDataset(data, targets, square, divide_by_two)
        actual = [dataset[i] for i in range(len(dataset))]
        expected = [(1, 1), (4, 2), (9, 3)]
        assert actual == expected
