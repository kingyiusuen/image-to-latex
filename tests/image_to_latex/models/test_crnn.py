import pytest
import torch

from image_to_latex.models.crnn import CRNN


class TestCRNN:
    @pytest.fixture()
    def config(self, tokenizer):
        return {
            "conv-dim": 32,
            "rnn-type": "RNN",
            "rnn-dim": 256,
            "rnn-layers": 2,
            "rnn-dropout": 0.4,
        }

    @pytest.fixture()
    def crnn(self, tokenizer, config):
        return CRNN(tokenizer, config)

    def test_config(self, crnn, config):
        assert crnn.config() == config

    def test_forward(self, crnn, images, num_samples, num_classes):
        output = crnn(images)
        expected_size = (
            num_samples,
            num_classes + 5,  # Add 5 for special tokens
            1 * 7,
        )
        assert output.size() == torch.Size(expected_size)
