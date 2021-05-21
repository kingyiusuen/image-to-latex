from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from image_to_latex.models import BaseModel
from image_to_latex.trainers import BaseTrainer


class ResnetTransformerTrainer(BaseTrainer):
    """Trainer for ResnetTransformer model."""

    def __init__(self, model: BaseModel, config: Dict[str, Any] = None) -> None:
        super().__init__(model, config)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_index)

    def training_step(self, batch: Sequence) -> torch.Tensor:
        """Training step."""
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        loss = self.criterion(logits, targets[:, 1:])
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Sequence) -> torch.Tensor:
        """Validation step."""
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        loss = self.criterion(logits, targets[:, 1:])
        return loss
