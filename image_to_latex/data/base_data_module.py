from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__()
        self.config = config if config is not None else {}
        self.batch_size = self.config.get("batch_size", BATCH_SIZE)
        self.num_workers = self.config.get("num_workers", NUM_WORKERS)

        self.transform = transforms.ToTensor()

        self.train_dataset: Dataset
        self.val_dataset: Dataset
        self.test_dataset: Dataset
        self.id2token: List[str]
        self.token2id: Dict[str, int]

    @classmethod
    def data_dirname(cls) -> Path:
        """Returns the directory to where data are stored."""
        return Path(__file__).resolve().parents[2] / "data"

    @staticmethod
    def add_to_argparse(parser):
        """Add arguments to a parser."""
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
        return parser

    def prepare_data(self) -> None:
        """Download data."""

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """Assign `torch Dataset` objects.

        When stage = "fit", should assign `torch Dataset` objects to
        self.train_dataset and self.val_dataset. When stage = "test", should
        assign a `torch Dataset` object to self.test_dataset.

        Args:
            stage: {"fit", "test", None}
        """

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
