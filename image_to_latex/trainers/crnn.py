from typing import Sequence

import torch
import torch.nn as nn

from image_to_latex.trainers import BaseTrainer
from image_to_latex.utils.misc import find_first


class CRNNTrainer(BaseTrainer):
    """Trainer for CRNN model."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = nn.CTCLoss(zero_infinity=True)

    def training_step(self, batch: Sequence) -> torch.Tensor:
        """Training step."""
        imgs, targets = batch
        log_probs = self.model(imgs)  # (B, num_classes, S)
        B, _, S = log_probs.shape
        log_probs_for_loss = log_probs.permute(2, 0, 1)  # (S, B, num_classes)
        input_lengths = torch.ones(B).type_as(log_probs_for_loss).int() * S
        target_lengths = find_first(targets, element=self.tokenizer.pad_index)
        loss = self.criterion(
            log_probs_for_loss, targets, input_lengths, target_lengths
        )
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Sequence) -> torch.Tensor:
        """Validation step."""
        imgs, targets = batch
        log_probs = self.model(imgs)
        B, _, S = log_probs.shape

        log_probs_for_loss = log_probs.permute(2, 0, 1)  # (S, B, num_classes)
        input_lengths = torch.ones(B).type_as(log_probs_for_loss).int() * S
        target_lengths = find_first(targets, element=self.tokenizer.pad_index)
        target_lengths = target_lengths.type_as(targets)
        loss = self.criterion(
            log_probs_for_loss, targets, input_lengths, target_lengths
        )
        return loss
