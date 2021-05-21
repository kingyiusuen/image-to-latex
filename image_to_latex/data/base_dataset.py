from typing import Callable, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """A base Dataset class.

    Args:
        data: (N, *) feature vector.
        targets: (N, *) target vector relative to data.
        transform: Feature transformation.
        target_transform: Target transformation.
    """

    def __init__(
        self,
        data: Union[Sequence, torch.Tensor, np.ndarray],
        targets: Union[Sequence, torch.Tensor, np.ndarray],
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        super().__init__()
        assert len(data) == len(targets)
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sample from the dataset at the given index."""
        datum, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            datum = self.transform(datum)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return datum, target
