import random
from typing import Dict, Generator, List

import torch
from torch.utils.data import Sampler

from image_to_latex.data.base_dataset import BaseDataset


class SameSizeBatchSampler(Sampler):
    """Sample images of the same size in the same batch.

    https://stackoverflow.com/questions/50663803/training-on-minibatches-of-varying-size

    Args:
    dataset: Dataset from which to load the data.
    batch_size: How many samples per batch to load.
    shuffle: set to `True` to have the data reshuffled at every epoch
    drop_last: set to `True` to drop the last incomplete batch, if the dataset
        size is not divisible by the batch size.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.buckets = _get_buckets(dataset)

    def __iter__(self) -> Generator[List[int], None, None]:
        """Iterate over indices.

        Returns:
            A list of sampled indices.
        """
        batch = []
        # Process buckets in random order
        sizes = list(self.buckets)
        if self.shuffle:
            sizes = random.sample(sizes, len(self.buckets))
        for size in sizes:
            # Process images in buckets in random order
            img_indices = self.buckets[size]
            if self.shuffle:
                img_indices = random.sample(img_indices, len(img_indices))
            for i in img_indices:
                batch.append(i)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            # Yield half-full batch before moving to next bucket
            # (if drop_last is True)
            if len(batch) > 0:
                if not self.drop_last:
                    yield batch
                batch = []

    def __len__(self) -> int:
        """Returns the number of batches we can iterate."""
        num_batches = 0
        for bucket in self.buckets.values():
            q, r = divmod(len(bucket), self.batch_size)
            num_batches += q
            if r > 0 and not self.drop_last:
                num_batches += 1
        return num_batches


def _get_buckets(dataset: BaseDataset) -> Dict[torch.Size, List[int]]:
    """Distribute indices into buckets.

    Images of the same size will be put in the same bucket.

    Args:
        dataset: Dataset from which to load the data.

    Returns:
        A dictionary that maps size to indices.
    """
    buckets: Dict[torch.Size, List[int]] = {}
    img: torch.Tensor
    for i, (img, _) in enumerate(dataset):  # type: ignore
        size = img.shape
        if size not in buckets:
            buckets[size] = []
        buckets[size].append(i)
    return buckets
