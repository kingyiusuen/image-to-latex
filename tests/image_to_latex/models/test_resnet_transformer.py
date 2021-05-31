import pytest
import torch

from image_to_latex.models.resnet_transformer import ResnetTransformer


class TestResnetTransformer:
    @pytest.fixture()
    def model_config(self):
        return {
            "model_name": "ResnetTransformer",
            "resnet_layers": 2,
            "tf_dim": 128,
            "tf_fc_dim": 256,
            "tf_nhead": 4,
            "tf_dropout": 0.4,
            "tf_layers": 4,
            "max_output_len": 30,
        }

    @pytest.fixture()
    def model(self, tokenizer, model_config):
        torch.manual_seed(0)
        model = ResnetTransformer(tokenizer, model_config)
        model.eval()
        return model

    def test_config(self, model, model_config):
        assert model.config() == model_config

    def test_greedy_and_beam_searches(self, model, train_dataloader):
        images, _ = next(iter(train_dataloader))
        gs_res = model._greedy_search(images, max_output_len=30)
        bs_res = model._beam_search(images, beam_width=1, max_output_len=30)
        assert torch.equal(gs_res, bs_res)
