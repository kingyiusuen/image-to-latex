from itertools import groupby
from typing import Any, Dict, List, Optional

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
MAX_OUTPUT_LENGTH = 152


class ConvReLU(nn.Module):
    """A convoluational operation followed by a ReLU operation."""

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

    References:
    Shi, B., Bai, X., & Yao, C. (2016). An end-to-end trainable neural network
    for image-based sequence recognition and its application to scene text
    recognition. IEEE transactions on pattern analysis and machine intelligence,
    39(11), 2298-2304. https://arxiv.org/abs/1507.05717.
    """

    def __init__(self, id2token: List[str], config: Dict[str, Any] = None) -> None:
        super().__init__(id2token, config)

        conv_dim = self.config.get("conv_dim", CONV_DIM)
        rnn_type = self.config.get("rnn_type", RNN_TYPE).upper()
        assert rnn_type in ["RNN", "LSTM", "GRU"]
        rnn_dim = self.config.get("rnn_dim", RNN_DIM)
        rnn_layers = self.config.get("rnn_layers", RNN_LAYERS)
        rnn_dropout = self.config.get("rnn_dropout", RNN_DROPOUT)

        self.cnn = nn.Sequential(
            ConvReLU(1, conv_dim, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            ConvReLU(conv_dim, conv_dim * 2, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            ConvReLU(conv_dim * 2, conv_dim * 4, kernel_size=3, padding=1),
            ConvReLU(conv_dim * 4, conv_dim * 4, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 1), stride=2),
            ConvReLU(conv_dim * 4, conv_dim * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim * 8),
            ConvReLU(conv_dim * 8, conv_dim * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim * 8),
            nn.MaxPool2d((2, 1), stride=2),
            ConvReLU(conv_dim * 8, conv_dim * 8, kernel_size=2, padding=0),
        )
        self.map_to_sequence = nn.Linear(conv_dim * 8, rnn_dim)
        rnn_class = import_class(f"torch.nn.{rnn_type}")
        self.rnn = rnn_class(
            input_size=rnn_dim,
            hidden_size=rnn_dim,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(rnn_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.cnn(x)
        B, C, H, W = x.size()
        x = x.view(B, C * H, W)  # (B, C * H, W)
        x = x.permute(0, 2, 1)  # (B, W, C * H)
        x = self.map_to_sequence(
            x
        )  # (B, W, rnn_dim) where S = W is the sequence length
        x = x.permute(1, 0, 2)  # (S, B, rnn_dim)
        x, _ = self.rnn(x)  # (S, B, rnn_dim * 2)

        # Sum up both directions of RNN:
        x = x.view(W, B, 2, -1).sum(dim=2)  # (S, B, rnn_dim)

        x = x.permute(1, 0, 2)  # (B, S, rnn_dim)
        x = self.fc(x)  # (B, S, num_classes)
        logits = x.permute(0, 2, 1)  # (B, num_classes, S)
        logprobs = torch.log_softmax(logits, dim=1)  # (B, num_classes, S)
        return logprobs

    def predict(
        self,
        x: torch.Tensor,
        max_output_length: Optional[int] = 200,
    ) -> torch.Tensor:
        """Make predictions at inference time."""
        if max_output_length is None:
            max_output_length = 200

        logprobs = self(x)
        B = logprobs.shape[0]
        argmax = logprobs.argmax(1)
        decoded = (
            torch.ones((B, max_output_length)).type_as(logprobs).int()
        ) * self.padding_index
        for i in range(B):
            seq = [b for b, _ in groupby(argmax[i].tolist()) if b != self.blank_index][
                :max_output_length
            ]
            for ii, char in enumerate(seq):
                decoded[i, ii] = char
        return decoded

    @staticmethod
    def add_to_argparse(parser):
        """Add arguments to a parser."""
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--rnn_type", type=str, default=RNN_TYPE)
        parser.add_argument("--rnn_dim", type=int, default=RNN_DIM)
        parser.add_argument("--rnn_layers", type=int, default=RNN_LAYERS)
        parser.add_argument("--rnn_dropout", type=int, default=RNN_DROPOUT)
        parser.add_argument("--max_output_length", type=int, default=MAX_OUTPUT_LENGTH)
        return parser
