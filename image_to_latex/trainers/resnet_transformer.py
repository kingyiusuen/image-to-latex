from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

import wandb
from image_to_latex.models import BaseModel
from image_to_latex.trainers import BaseTrainer


class ResnetTransformerTrainer(BaseTrainer):
    """Trainer for ResnetTransformer model."""

    def __init__(
        self,
        model: BaseModel,
        config: Dict[str, Any],
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
        save_best_model: bool = False,
    ) -> None:
        super().__init__(model, config, wandb_run, save_best_model)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_index
        )

    def training_step(self, batch: Sequence) -> torch.Tensor:
        """Training step."""
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        loss = self.criterion(logits, targets[:, 1:])
        if self.wandb_run:
            wandb.log({"train_loss": loss.item()})
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Sequence) -> torch.Tensor:
        """Validation step."""
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        loss = self.criterion(logits, targets[:, 1:])
        if self.wandb_run:
            wandb.log({"val_loss": loss.item()})
        return loss
