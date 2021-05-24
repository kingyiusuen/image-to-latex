import numpy as np
import pytest

from image_to_latex.data.fake_data import FakeData


class TestFakeData:
    @pytest.fixture
    def fake_data(self):
        np.random.seed(32)
        config = {
            "num-samples": 20,
            "image-height": 16,
            "image-width": 16,
            "num-classes": 4,
            "max-seq-len": 8,
        }
        return FakeData(0, 0, config)

    def test_create_datasets(self, fake_data):
        fake_data.create_datasets()
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
                assert len(target) == (8 + 2)  # Add 2 for sos and eos tokens
                vocab |= {x.item() for x in target}
        assert len(vocab) == (4 + 3)  # Add 3 for sos, eos, pad tokens
