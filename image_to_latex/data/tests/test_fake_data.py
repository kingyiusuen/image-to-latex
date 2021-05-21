import numpy as np
import pytest

from image_to_latex.data.fake_data import FakeData


class TestFakeData:
    @pytest.fixture
    def fake_data(self):
        np.random.seed(32)
        config = {
            "num_samples": 20,
            "image_height": 16,
            "image_width": 16,
            "num_classes": 4,
            "seq_length": 8,
        }
        return FakeData(config)

    def test_setup(self, fake_data):
        fake_data.setup()
        vocab = set()
        for split in ["train", "val", "test"]:
            dataset = getattr(fake_data, f"{split}_dataset")
            if split == "train":
                assert len(dataset) == 10
            else:
                assert len(dataset) == 5
            for item in dataset:
                assert len(item) == 2
                img, target = item
                assert img.shape == (1, 16, 16)
                assert target.shape == (8,)
                vocab |= set([x for x in target])
        assert len(vocab) == 4
