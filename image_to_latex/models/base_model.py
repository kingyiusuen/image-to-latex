from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from image_to_latex.utils.data import Tokenizer


class BaseModel(nn.Module, ABC):
    """Base class for a model."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        self.blk_index = self.tokenizer.blk_index
        self.sos_index = self.tokenizer.sos_index
        self.eos_index = self.tokenizer.eos_index
        self.pad_index = self.tokenizer.pad_index
        self.num_classes = len(self.tokenizer)

    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Returns important configuration for reproducibility."""

    @abstractmethod
    def predict(
        self,
        x: torch.Tensor,
        max_output_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Make predictions at inference time."""
