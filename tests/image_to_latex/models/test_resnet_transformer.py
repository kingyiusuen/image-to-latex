import pytest

from image_to_latex.models.resnet_transformer import ResnetTransformer


class TestCRNN:
    @pytest.fixture()
    def config(self, tokenizer):
        return {
            "resnet-layers": 2,
            "tf-dim": 128,
            "tf-fc-dim": 256,
            "tf-nhead": 4,
            "tf-dropout": 0.4,
            "tf-layers": 4,
            "max-output-len": 250,
        }

    @pytest.fixture()
    def resnet_transformer(self, tokenizer, config):
        return ResnetTransformer(tokenizer, config)

    def test_config(self, resnet_transformer, config):
        assert resnet_transformer.config() == config
