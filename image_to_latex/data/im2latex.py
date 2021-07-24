import random
from pathlib import Path
from typing import Optional

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .utils import BaseDataset, Tokenizer, get_all_formulas, get_split


class Im2Latex(LightningDataModule):
    """Data processing for the Im2Latex-100K dataset.

    Args:
        batch_size: The number of samples per batch.
        num_workers: The number of subprocesses to use for data loading.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory
            before returning them.
    """

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_dirname = Path(__file__).resolve().parents[2] / "data"
        self.vocab_file = Path(__file__).resolve().parent / "vocab.json"
        formula_file = self.data_dirname / "im2latex_formulas.norm.new.lst"
        if not formula_file.is_file():
            raise FileNotFoundError("Did you run scripts/prepare_data.py?")
        self.all_formulas = get_all_formulas(formula_file)
        self.transform = {
            "train": A.Compose(
                [
                    A.Affine(scale=(0.6, 1.0), rotate=(-1, 1), cval=255, p=0.5),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=(1, 1), p=0.5),
                    ToTensorV2(),
                ]
            ),
            "val/test": ToTensorV2(),
        }

    @property
    def processed_images_dirname(self):
        return self.data_dirname / "formula_images_processed"

    def setup(self, stage: Optional[str] = None) -> None:
        """Load images and formulas, and assign them to a `torch Dataset`.

        `self.train_dataset`, `self.val_dataset` and `self.test_dataset` will
        be assigned after this method is called.
        """
        self.tokenizer = Tokenizer.load(self.vocab_file)

        if stage in ("fit", None):
            train_image_names, train_formulas = get_split(
                self.all_formulas,
                self.data_dirname / "im2latex_train_filter.lst",
            )
            self.train_dataset = BaseDataset(
                self.processed_images_dirname,
                image_filenames=train_image_names,
                formulas=train_formulas,
                transform=self.transform["train"],
            )

            val_image_names, val_formulas = get_split(
                self.all_formulas,
                self.data_dirname / "im2latex_validate_filter.lst",
            )
            self.val_dataset = BaseDataset(
                self.processed_images_dirname,
                image_filenames=val_image_names,
                formulas=val_formulas,
                transform=self.transform["val/test"],
            )

        if stage in ("test", None):
            test_image_names, test_formulas = get_split(
                self.all_formulas,
                self.data_dirname / "im2latex_test_filter.lst",
            )
            self.test_dataset = BaseDataset(
                self.processed_images_dirname,
                image_filenames=test_image_names,
                formulas=test_formulas,
                transform=self.transform["val/test"],
            )

    def collate_fn(self, batch):
        images, formulas = zip(*batch)
        B = len(images)
        max_H = max(image.shape[1] for image in images)
        max_W = max(image.shape[2] for image in images)
        max_length = max(len(formula) for formula in formulas)
        padded_images = torch.zeros((B, 1, max_H, max_W))
        batched_indices = torch.zeros((B, max_length + 2), dtype=torch.long)
        for i in range(B):
            H, W = images[i].shape[1], images[i].shape[2]
            y, x = random.randint(0, max_H - H), random.randint(0, max_W - W)
            padded_images[i, :, y : y + H, x : x + W] = images[i]
            indices = self.tokenizer.encode(formulas[i])
            batched_indices[i, : len(indices)] = torch.tensor(indices, dtype=torch.long)
        return padded_images, batched_indices

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
