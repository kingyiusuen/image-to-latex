from typing import Sequence

import torch
import torch.nn as nn

from image_to_latex.trainers import BaseTrainer


class ResnetTransformerTrainer(BaseTrainer):
    """Trainer for ResnetTransformer model."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_index,
            reduction="sum",
        )

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
