import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from image_to_latex.data.base_data_module import BaseDataModule
from image_to_latex.data.base_dataset import BaseDataset
from image_to_latex.data.same_size_batch_sampler import SameSizeBatchSampler
from image_to_latex.utils.data import convert_strings_to_labels
from image_to_latex.utils.misc import (
    download_url,
    extract_tar_file,
    find_max_length,
    verify_sha256
)


DATA_DIRNAME = BaseDataModule.data_dirname()
FORMULA_FILENAME = DATA_DIRNAME / "im2latex_formulas.norm.lst"
VOCAB_FILENAME = DATA_DIRNAME / "vocab.json"


class Im2Latex(BaseDataModule):
    """Data processing for the Im2Latex-100K dataset.

    Attributes:
        token2id: A dictionary that maps tokens to indices. Can be created by
            calling `build_vocab`.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config)
        self.token2id: Dict[str, int]

    def prepare_data(self) -> None:
        """Download the dataset and save to disk."""
        DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        cur_dir = os.getcwd()
        os.chdir(DATA_DIRNAME)
        with open(DATA_DIRNAME / "metadata.json", "r") as f:
            metadata = json.load(f)
        for filename, url, sha256 in metadata:
            # No need to download the file if it already exists in the data
            # directory
            if Path(filename).is_file():
                continue
            download_url(url, filename)
            verify_sha256(filename, sha256)
            if filename == "formula_images_processed.tar.gz":
                extract_tar_file(filename)
        os.chdir(cur_dir)

    def setup(self, stage: str = None) -> None:
        """Load images and formulas, and assign them to a `torch Dataset`."""

        def _get_dataset(
            img_names: Sequence[str], formulas: Sequence[Sequence[str]], seq_length: int
        ) -> Dataset:
            images = [
                Image.open(_img_filename(img_name)).convert("L")
                for img_name in img_names
            ]
            targets = convert_strings_to_labels(formulas, self.token2id, seq_length)
            return BaseDataset(images, targets, self.transform)

        if not hasattr(self, "token2id"):
            raise RuntimeError(
                "Should call `build_vocab` first to create a token-to-index "
                "mapping, so that target labels of the dataset(s) can be "
                "generated accordingly."
            )

        print("Loading dataset(s)...")

        if stage == "fit" or stage is None:
            formulas = get_formulas()
            train_img_names, train_formula_indices = load_split_file("train")
            val_img_names, val_formula_indices = load_split_file("val")
            train_formulas = filter_formulas(formulas, train_formula_indices)
            val_formulas = filter_formulas(formulas, val_formula_indices)
            seq_length = max(find_max_length(train_formulas), find_max_length(val_formulas))
            seq_length += 2  # Add two for start token and end token
            self.train_dataset = _get_dataset(
                train_img_names, train_formulas, seq_length
            )
            self.val_dataset = _get_dataset(val_img_names, val_formulas, seq_length)

        if stage == "test" or stage is None:
            formulas = get_formulas()
            test_img_names, test_formula_indices = load_split_file("test")
            test_formulas = filter_formulas(formulas, test_formula_indices)
            seq_length = find_max_length(test_formulas) + 2
            # Filter out formulas that have zero length
            test_img_names_ = []
            test_formulas_ = []
            for img_name, formula in zip(test_img_names, test_formulas):
                if len(formula) > 0:
                    test_img_names_.append(img_name)
                    test_formulas_.append(formula)
            self.test_dataset = _get_dataset(
                test_img_names_, test_formulas_, seq_length
            )

    def get_dataloader(self, split: str) -> DataLoader:
        """Returns a `torch Dataloader` object."""
        assert split in ["train", "val", "test"]
        print(f"Preparing {split}_dataloader...")
        dataset = getattr(self, f"{split}_dataset")
        batch_sampler = SameSizeBatchSampler(
            dataset, batch_size=self.batch_size, shuffle=(split == "train")
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return dataloader

    def build_vocab(self, min_count: int = 2) -> None:
        """Create a mapping from tokens to indices.

        Args:
            min_count: Tokens that appear fewer than `min_count` will not be
                included in the mapping.
        """
        if VOCAB_FILENAME.is_file():
            with open(VOCAB_FILENAME, "r") as f:
                self.token2id = json.load(f)
            return

        # Get the tokens from the formulas in the training dataset
        self.prepare_data()
        formulas = get_formulas()
        _, train_formula_indices = load_split_file("train")
        train_formulas = filter_formulas(formulas, train_formula_indices)

        # Count the frequency of each token
        counter: Dict[str, int] = {}
        for formula in train_formulas:
            for token in formula:
                counter[token] = counter.get(token, 0) + 1

        # Remove tokens that show up fewer than `min_count` times
        train_tokens = []
        for token, count in counter.items():
            if count >= min_count:
                train_tokens.append(token)

        # Create a mapping from tokens to indices
        # <NIL> blank token (for CTC) at index 0
        # <BOS> begin-of-sequence token at index 1
        # <EOS> end-of-sequence token at index 2
        # <PAD> padding token at index 3
        # <UNK> unkown token at index 4
        special_tokens = ["<NIL>", "<BOS>", "<EOS>", "<PAD>", "<UNK>"]
        self.id2token = special_tokens + train_tokens
        self.token2id = {token: i for i, token in enumerate(self.id2token)}

        # Save the mapping
        with open(VOCAB_FILENAME, "w") as f:
            json.dump(self.token2id, f)


def get_formulas() -> List[List[str]]:
    """Returns all the formulas in the formula file."""
    with open(FORMULA_FILENAME, "r") as f:
        formulas = [formula.strip("\n").split() for formula in f.readlines()]
    return formulas


def load_split_file(split: str) -> Tuple[List[str], List[int]]:
    """Load image names and formula indices from a split file."""
    img_names = []
    formula_indices = []
    with open(_split_filename(split), "r") as f:
        for line in f:
            img_name, formula_idx = line.strip("\n").split()
            img_names.append(img_name)
            formula_indices.append(int(formula_idx))
    return img_names, formula_indices


def filter_formulas(
    formulas: List[List[str]], formula_indices: List[int]
) -> List[List[str]]:
    """Filter formulas by indices."""
    return [formulas[idx] for idx in formula_indices]


def _split_filename(split: str) -> Path:
    """Returns the path to a split file."""
    if split == "val":
        split = "validate"
    return DATA_DIRNAME / f"im2latex_{split}_filter.lst"


def _img_filename(img_name: str) -> Path:
    """Returns the path to an image."""
    return DATA_DIRNAME / "formula_images_processed" / img_name
