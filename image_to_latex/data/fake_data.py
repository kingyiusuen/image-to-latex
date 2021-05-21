from typing import Any, Dict

import numpy as np
from torch.utils.data import random_split

from image_to_latex.data.base_data_module import BaseDataModule
from image_to_latex.data.base_dataset import BaseDataset


NUM_SAMPLES = 256
IMAGE_LEN = 32
NUM_CLASSES = 10
SEQ_LENGTH = 30


class FakeData(BaseDataModule):
    """Fake dataset for testing/debugging."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config)
        self.num_samples = self.config.get("num_samples", NUM_SAMPLES)
        self.image_height = self.config.get("image_height", IMAGE_LEN)
        self.image_width = self.config.get("image_width", IMAGE_LEN)
        self.num_classes = self.config.get("num_classes", NUM_CLASSES)
        self.seq_length = self.config.get("seq_length", SEQ_LENGTH)

    @staticmethod
    def add_to_argparse(parser):
        """Add arguments to a parser."""
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
        parser.add_argument("--image_height", type=int, default=IMAGE_LEN)
        parser.add_argument("--image_width", type=int, default=IMAGE_LEN)
        parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
        parser.add_argument("--seq_length", type=int, default=SEQ_LENGTH)
        return parser

    def setup(self, stage: str = None) -> None:
        """Generate fake data."""
        images = generate_random_images(
            self.num_samples, self.image_height, self.image_width
        )
        labels = generate_random_labels(
            self.num_samples, self.num_classes, self.seq_length
        )
        fake_dataset = BaseDataset(images, labels, self.transform)
        val_size = int(self.num_samples * 0.25)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset=fake_dataset,
            lengths=[self.num_samples - 2 * val_size, val_size, val_size],
        )

    def build_vocab(self) -> None:
        """Create a mapping from tokens to indices."""
        special_tokens = ["<NIL>", "<BOS>", "<EOS>", "<PAD>", "<UNK>"]
        tokens = [str(i) for i in range(self.num_classes)]
        self.id2token = special_tokens + tokens
        self.token2id = {token: i for i, token in enumerate(self.id2token)}


def generate_random_images(
    num_samples: int, image_height: int, image_width: int
) -> np.ndarray:
    """Generate random images."""
    size = (num_samples, image_height, image_width, 1)
    X = np.random.rand(*size)
    X = X.astype(np.float32)
    return X


def generate_random_labels(
    num_samples: int, num_classes: int, seq_length: int
) -> np.ndarray:
    """Generate random sequences."""
    size = (num_samples, seq_length)
    Y = np.random.randint(low=0, high=num_classes, size=size)
    return Y
