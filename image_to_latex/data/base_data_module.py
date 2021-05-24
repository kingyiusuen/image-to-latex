from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from image_to_latex.utils.data import Tokenizer


BATCH_SIZE = 32
NUM_WORKERS = 0


class BaseDataModule(ABC):
    """Handle data processing.

    A datamodule encapsulates the steps involved in data processing, such as
    downloading datasets, loading as a `torch Dataset` objects, applying
    transformation, and wraping inside a DataLoader.

    Inspired by Pytorch Lightning:
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.batch_size = config.get("batch-size", BATCH_SIZE)
        self.num_workers = config.get("num-workers", NUM_WORKERS)

        self.transform = transforms.ToTensor()

        self.train_dataset: Dataset
        self.val_dataset: Dataset
        self.test_dataset: Dataset
        self.tokenizer: Tokenizer

    @classmethod
    def data_dirname(cls) -> Path:
        """Returns the directory to where data are stored."""
        return Path(__file__).resolve().parents[2] / "data"

    def config(self) -> Dict[str, Any]:
        """Returns important configuration for reproducibility."""

    def prepare_data(self) -> None:
        """Download data."""

    @abstractmethod
    def create_datasets(self) -> None:
        """Assign `torch Dataset` objects."""

    def get_dataloader(self, split: str) -> DataLoader:
        """Returns a `torch Dataloader` object."""
        assert split in ["train", "val", "test"]
        print(f"Preparing {split}_dataloader...")
        dataloader = DataLoader(
            getattr(self, f"{split}_dataset"),
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return dataloader
