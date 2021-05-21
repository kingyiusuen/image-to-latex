from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Base class for a model."""

    def __init__(
        self,
        id2token: List[str],
        config: Dict[str, Any] = None
    ) -> None:
        super().__init__()
        self.id2token = id2token
        self.config = config if config is not None else {}

        self.token2id = {token: i for i, token in enumerate(self.id2token)}
        self.blank_index = self.token2id["<NIL>"]
        self.start_index = self.token2id["<BOS>"]
        self.end_index = self.token2id["<EOS>"]
        self.padding_index = self.token2id["<PAD>"]
        self.num_classes = len(self.id2token)

    @abstractmethod
    def predict(
        self, x: torch.Tensor, max_output_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Make predictions at inference time."""
