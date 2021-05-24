from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import random_split

from image_to_latex.data.base_data_module import BaseDataModule
from image_to_latex.data.base_dataset import BaseDataset
from image_to_latex.utils.data import Tokenizer


NUM_SAMPLES = 256
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 128
NUM_CLASSES = 10
MAX_SEQ_LEN = 10


class FakeData(BaseDataModule):
    """Fake dataset for testing/debugging."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(batch_size, num_workers, config)
        self.num_samples = config.get("num-samples", NUM_SAMPLES)
        self.image_height = config.get("image-height", IMAGE_HEIGHT)
        self.image_width = config.get("image-width", IMAGE_WIDTH)
        self.num_classes = config.get("num-classes", NUM_CLASSES)
        self.max_seq_len = config.get("max-seq-len", MAX_SEQ_LEN)
        self.tokenizer = Tokenizer()

    def config(self) -> Dict[str, Any]:
        """Returns important configuration for reproducibility."""
        return {
            "batch-size": self.batch_size,
            "num-workers": self.num_workers,
            "num-samples": self.num_samples,
            "image-height": self.image_height,
            "image-width": self.image_width,
            "num-classes": self.num_classes,
            "max-seq-len": self.max_seq_len,
        }

    def create_datasets(self) -> None:
        """Generate fake data."""
        images = generate_random_images(
            self.num_samples, self.image_height, self.image_width
        )
        seqs = generate_random_seqs(
            self.num_samples, self.num_classes, self.max_seq_len
        )
        self.tokenizer.build(seqs, min_count=1)
        targets = self.tokenizer.index(
            seqs, add_sos=True, add_eos=True, pad_to=(self.max_seq_len + 2)
        )
        fake_dataset = BaseDataset(
            images, torch.IntTensor(targets), self.transform
        )
        val_size = int(self.num_samples * 0.25)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset=fake_dataset,
            lengths=[self.num_samples - 2 * val_size, val_size, val_size],
        )


def generate_random_images(
    num_samples: int, image_height: int, image_width: int
) -> np.ndarray:
    """Generate random images."""
    size = (num_samples, image_height, image_width, 1)
    images = np.random.rand(*size)
    images = images.astype(np.float32)
    return images


def generate_random_seqs(
    num_samples: int, num_classes: int, max_seq_len: int
) -> List[np.ndarray]:
    """Generate random sequences with variable lengths."""
    seqs = []
    for _ in range(num_samples):
        seq_len = np.random.randint(1, max_seq_len + 1)
        seq = np.random.randint(low=0, high=num_classes, size=seq_len)
        seqs.append(seq)
    return seqs
