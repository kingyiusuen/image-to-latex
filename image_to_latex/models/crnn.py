from itertools import groupby
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_to_latex.models import BaseModel
from image_to_latex.utils.misc import import_class


CONV_DIM = 64
RNN_TYPE = "LSTM"
RNN_DIM = 256
RNN_LAYERS = 2
RNN_DROPOUT = 0.4
MAX_OUTPUT_LEN = 250


class ConvReLU(nn.Module):
    """A convoluation followed by a ReLU operation."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        """Forward pass."""
        x = self.conv(x)
        x = F.relu(x)
        return x


class CRNN(BaseModel):
    """Implementation of a Convoluational Recurrent Neural Network.

    Note that this model requires the images to have the same size.

    References:
    Shi, B., Bai, X., & Yao, C. (2016). An end-to-end trainable neural network
    for image-based sequence recognition and its application to scene text
    recognition. IEEE transactions on pattern analysis and machine
    intelligence, 39(11), 2298-2304. https://arxiv.org/abs/1507.05717.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv_dim = self.args.get("conv_dim", CONV_DIM)
        self.rnn_type = self.args.get("rnn_type", RNN_TYPE).upper()
        assert self.rnn_type in ["RNN", "LSTM", "GRU"]
        self.rnn_dim = self.args.get("rnn_dim", RNN_DIM)
        self.rnn_layers = self.args.get("rnn_layers", RNN_LAYERS)
        self.rnn_dropout = self.args.get("rnn_dropout", RNN_DROPOUT)

        self.cnn = nn.Sequential(
            ConvReLU(1, self.conv_dim, 3, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            ConvReLU(self.conv_dim, self.conv_dim * 2, 3, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            ConvReLU(self.conv_dim * 2, self.conv_dim * 4, 3, padding=1),
            ConvReLU(self.conv_dim * 4, self.conv_dim * 4, 3, padding=1),
            nn.MaxPool2d((2, 1), stride=2),
            ConvReLU(self.conv_dim * 4, self.conv_dim * 8, 3, padding=1),
            nn.BatchNorm2d(self.conv_dim * 8),
            ConvReLU(self.conv_dim * 8, self.conv_dim * 8, 3, padding=1),
            nn.BatchNorm2d(self.conv_dim * 8),
            nn.MaxPool2d((2, 1), stride=2),
            ConvReLU(self.conv_dim * 8, self.conv_dim * 8, 2, padding=0),
        )
        self.map_to_sequence = nn.Linear(self.conv_dim * 8, self.rnn_dim)
        rnn_class = import_class(f"torch.nn.{self.rnn_type}")
        self.rnn = rnn_class(
            input_size=self.rnn_dim,
            hidden_size=self.rnn_dim,
            num_layers=self.rnn_layers,
            dropout=self.rnn_dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(self.rnn_dim, self.num_classes)

    def config(self) -> Dict[str, Any]:
        """Returns important configuration for reproducibility."""
        return {
            "model_name": "CRNN",
            "conv_dim": self.conv_dim,
            "rnn_type": self.rnn_type,
            "rnn_dim": self.rnn_dim,
            "rnn_layers": self.rnn_layers,
            "rnn_dropout": self.rnn_dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, _C, _H, _W)

        Returns:
            (B, num_classes, S)
        """
        x = self.cnn(x)
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = x.contiguous().view(B, H * W, C)  # (B, H * W, C)

        S = H * W
        x = self.map_to_sequence(x)  # (B, S, rnn_dim)
        x = x.permute(1, 0, 2)  # (S, B, rnn_dim)
        x, _ = self.rnn(x)  # (S, B, rnn_dim * 2)

        # Sum up both directions of RNN:
        x = x.view(S, B, 2, -1).sum(dim=2)  # (S, B, rnn_dim)

        x = x.permute(1, 0, 2)  # (B, S, rnn_dim)
        x = self.fc(x)  # (B, S, num_classes)
        logits = x.permute(0, 2, 1)  # (B, num_classes, S)
        log_probs = torch.log_softmax(logits, dim=1)  # (B, num_classes, S)
        return log_probs

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Make predictions at inference time."""
        max_output_len = kwargs.get("max_output_len", MAX_OUTPUT_LEN)

        log_probs = self(x)
        B = log_probs.shape[0]
        argmax = log_probs.argmax(1)
        decoded = (
            torch.ones((B, max_output_len)).type_as(log_probs).int()
        ) * self.pad_index
        for i in range(B):
            seq = [
                b
                for b, _ in groupby(argmax[i].tolist())
                if b != self.blk_index
            ]
            seq = seq[:max_output_len]
            for ii, char in enumerate(seq):
                decoded[i, ii] = char
        return decoded
