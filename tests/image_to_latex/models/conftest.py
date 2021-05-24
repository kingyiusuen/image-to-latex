import pytest
import torch

from image_to_latex.utils.data import Tokenizer


@pytest.fixture
def num_samples():
    return 32


@pytest.fixture
def image_height():
    return 32


@pytest.fixture
def image_width():
    return 128


@pytest.fixture
def num_classes():
    return 10


@pytest.fixture
def max_seq_len():
    return 20


@pytest.fixture
def images(num_samples, image_height, image_width):
    return torch.rand(num_samples, 1, image_height, image_width)


@pytest.fixture
def tokenizer(num_classes):
    seqs = [[i for i in range(num_classes)]]
    tokenizer = Tokenizer()
    tokenizer.build(seqs, min_count=1)
    return tokenizer
