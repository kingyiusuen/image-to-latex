from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from image_to_latex.utils.data import Tokenizer


class BaseDataModule(ABC):
    """Handle data processing.

    A datamodule encapsulates the steps involved in data processing, such as
    downloading datasets, loading as a `torch Dataset` objects, applying
    transformation, and wraping inside a DataLoader.

    Inspired by Pytorch Lightning:
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args if args else {}

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
