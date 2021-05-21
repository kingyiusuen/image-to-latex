from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from image_to_latex.models import BaseModel
from image_to_latex.trainers import BaseTrainer
from image_to_latex.utils.misc import find_first


class CRNNTrainer(BaseTrainer):
    """Trainer for CRNN model."""

    def __init__(self, model: BaseModel, config: Dict[str, Any] = None) -> None:
        super().__init__(model, config)
        self.criterion = nn.CTCLoss(zero_infinity=True)

    def training_step(self, batch: Sequence) -> torch.Tensor:
        """Training step."""
        imgs, targets = batch
        logprobs = self.model(imgs)  # (B, num_classes, S)
        B, _, S = logprobs.shape

        logprobs_for_loss = logprobs.permute(2, 0, 1)  # (S, B, num_classes)
        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S
        target_lengths = find_first(targets, element=self.model.padding_index)
        target_lengths = target_lengths.type_as(targets)
        loss = self.criterion(logprobs_for_loss, targets, input_lengths, target_lengths)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Sequence) -> torch.Tensor:
        """Validation step."""
        imgs, targets = batch
        logprobs = self.model(imgs)
        B, _, S = logprobs.shape

        logprobs_for_loss = logprobs.permute(2, 0, 1)  # (S, B, num_classes)
        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S
        target_lengths = find_first(targets, element=self.model.padding_index)
        target_lengths = target_lengths.type_as(targets)
        loss = self.criterion(logprobs_for_loss, targets, input_lengths, target_lengths)
        return loss
