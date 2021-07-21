from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from ..data.utils import Tokenizer
from ..models import ResNetTransformer
from .metrics import EditDistance


class LitResNetTransformer(LightningModule):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        dropout: float,
        num_decoder_layers: int,
        max_output_len: int,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        milestones: List[int] = [5],
        gamma: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma

        vocab_file = Path(__file__).resolve().parents[1] / "data" / "vocab.json"
        self.tokenizer = Tokenizer.load(vocab_file)
        self.model = ResNetTransformer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
            max_output_len=max_output_len,
            sos_index=self.tokenizer.sos_index,
            eos_index=self.tokenizer.eos_index,
            pad_index=self.tokenizer.pad_index,
            num_classes=len(self.tokenizer),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_edit_distance = EditDistance(self.tokenizer.special_tokens)
        self.test_edit_distance = EditDistance(self.tokenizer.special_tokens)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        self.log("val/loss", loss)

        preds = self.model.predict(imgs)
        edit_distance = self.val_edit_distance(preds, targets)
        self.log("val/edit_distance", edit_distance)

    def test_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self.model.predict(imgs)
        edit_distance = self.test_edit_distance(preds, targets)
        self.log("test/edit_distance", edit_distance)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]
