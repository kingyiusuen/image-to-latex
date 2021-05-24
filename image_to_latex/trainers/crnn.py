from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

import wandb
from image_to_latex.models import BaseModel
from image_to_latex.trainers import BaseTrainer
from image_to_latex.utils.misc import find_first


class CRNNTrainer(BaseTrainer):
    """Trainer for CRNN model."""

    def __init__(
        self,
        model: BaseModel,
        config: Dict[str, Any],
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
        save_best_model: bool = False,
    ) -> None:
        super().__init__(model, config, wandb_run, save_best_model)
        self.criterion = nn.CTCLoss(zero_infinity=True)

    def training_step(self, batch: Sequence) -> torch.Tensor:
        """Training step."""
        imgs, targets = batch
        logprobs = self.model(imgs)  # (B, num_classes, S)
        B, _, S = logprobs.shape
        logprobs_for_loss = logprobs.permute(2, 0, 1)  # (S, B, num_classes)
        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S
        target_lengths = find_first(targets, element=self.tokenizer.pad_index)
        loss = self.criterion(
            logprobs_for_loss, targets, input_lengths, target_lengths
        )
        if self.wandb_run:
            wandb.log({"train_loss": loss.item()})
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Sequence) -> torch.Tensor:
        """Validation step."""
        imgs, targets = batch
        logprobs = self.model(imgs)
        B, _, S = logprobs.shape

        logprobs_for_loss = logprobs.permute(2, 0, 1)  # (S, B, num_classes)
        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S
        target_lengths = find_first(targets, element=self.tokenizer.pad_index)
        target_lengths = target_lengths.type_as(targets)
        loss = self.criterion(
            logprobs_for_loss, targets, input_lengths, target_lengths
        )
        if self.wandb_run:
            wandb.log({"val_loss": loss.item()})
        return loss
