import pytest
import torch

from image_to_latex.models.crnn import CRNN


class TestCRNN:
    @pytest.fixture()
    def model_config(self):
        return {
            "model_name": "CRNN",
            "conv_dim": 32,
            "rnn_type": "RNN",
            "rnn_dim": 256,
            "rnn_layers": 2,
            "rnn_dropout": 0.4,
        }

    @pytest.fixture()
    def model(self, tokenizer, model_config):
        model = CRNN(tokenizer, model_config)
        model.eval()
        return model

    def test_config(self, model, model_config):
        assert model.config() == model_config

    def test_forward(self, model, train_dataloader, tokenizer):
        images, _ = next(iter(train_dataloader))
        num_samples = len(images)
        num_classes = len(tokenizer)
        output = model(images)
        expected_size = (
            num_samples,
            num_classes,
            1 * 7,
        )
        assert output.size() == torch.Size(expected_size)
