import pytest
import torch

from image_to_latex.utils.data import (
    convert_labels_to_strings,
    convert_strings_to_labels
)


class TestConvertStringsToLabels:
    @pytest.fixture
    def token2id(self):
        return {
            "<NIL>": 0, "<BOS>": 1, "<EOS>": 2, "<PAD>": 3, "<UNK>": 4,
            r"\alpha": 5, r"\beta": 6
        }

    @pytest.fixture
    def seq_length(self):
        return 5

    def test_known_tokens(self, token2id, seq_length):
        strings = [[r"\alpha", r"\beta", r"\alpha"]]
        actual = convert_strings_to_labels(strings, token2id, seq_length)
        expected = torch.LongTensor([[1, 5, 6, 5, 2]])
        assert torch.equal(actual, expected)

    def test_unknown_tokens(self, token2id, seq_length):
        strings = [[r"\zeta"]]
        actual = convert_strings_to_labels(strings, token2id, seq_length)
        expected = torch.LongTensor([[1, 4, 2, 3, 3]])
        assert torch.equal(actual, expected)

    def test_two_seqs(self, token2id, seq_length):
        strings = [[r"\alpha", r"\alpha", r"\alpha"], [r"\beta", r"\beta"]]
        actual = convert_strings_to_labels(strings, token2id, seq_length)
        expected = torch.LongTensor([[1, 5, 5, 5, 2], [1, 6, 6, 2, 3]])
        assert torch.equal(actual, expected)


class TestConvertLabelsToStrings:
    @pytest.fixture
    def id2token(self):
        return {
            1: "<BOS>", 2: "<EOS>", 3: "<PAD>", 4: "<UNK>",
            5: r"\alpha", 6: r"\beta"
        }

    def test_known_tokens(self, id2token):
        labels = torch.LongTensor([[1, 5, 6, 2, 3]])
        actual = convert_labels_to_strings(labels, id2token)
        expected = [[r"\alpha", r"\beta"]]
        assert actual == expected

    def test_ignored_tokens(self, id2token):
        labels = torch.LongTensor([[1, 1, 4, 5, 3, 2, 3, 3]])
        actual = convert_labels_to_strings(
            labels,
            id2token,
            ignored_tokens=["<BOS>", "<PAD>", "<UNK>"]
        )
        expected = [[r"\alpha"]]
        assert actual == expected

    def test_two_end_tokens(self, id2token):
        labels = torch.LongTensor([[1, 5, 6, 2, 3, 3, 2]])
        actual = convert_labels_to_strings(labels, id2token)
        expected = [[r"\alpha", r"\beta"]]
        assert actual == expected

    def test_two_seqs(self, id2token):
        labels = torch.LongTensor([[1, 5, 6, 2, 3], [1, 6, 2, 3, 3]])
        actual = convert_labels_to_strings(labels, id2token)
        expected = [[r"\alpha", r"\beta"], [r"\beta"]]
        assert actual == expected
