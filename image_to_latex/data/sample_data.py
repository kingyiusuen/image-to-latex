import json
from typing import Any, Dict

import torch
from PIL import Image

from image_to_latex.data import BaseDataModule, BaseDataset
from image_to_latex.utils.data import Tokenizer
from image_to_latex.utils.misc import find_max_length


DATA_DIRNAME = BaseDataModule.data_dirname()
SAMPLE_DATA_DIRNAME = DATA_DIRNAME / "sample_data"


class SampleData(BaseDataModule):
    """Sample Dataset used for testing/debugging.

    The same data is used for train/val/test to save time.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = Tokenizer()

    def config(self) -> Dict[str, Any]:
        """Returns important configuration for reproducibility."""
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }

    def create_datasets(self) -> None:
        with open(SAMPLE_DATA_DIRNAME / "sample_data.json") as f:
            unittest_data = json.load(f)
        images = []
        formulas = []
        for entry in unittest_data:
            filename = entry["filename"]
            formula = entry["formula"].split()
            image = Image.open(SAMPLE_DATA_DIRNAME / filename).convert("L")
            images.append(image)
            formulas.append(formula)
        self.tokenizer.build(formulas)
        targets = self.tokenizer.index(
            formulas,
            add_sos=True,
            add_eos=True,
            pad_to=(find_max_length(formulas) + 2),
        )
        self.train_dataset = BaseDataset(
            data=images,
            targets=torch.LongTensor(targets),
            transform=self.transform,
        )
        self.val_dataset = self.train_dataset
        self.test_dataset = self.train_dataset
