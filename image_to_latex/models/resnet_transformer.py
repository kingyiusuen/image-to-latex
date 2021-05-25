import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchvision.models

from image_to_latex.models import (
    BaseModel,
    PositionalEncoding1D,
    PositionalEncoding2D,
)
from image_to_latex.models.beam_search import (
    BeamSearchCandidate,
    TopKPriorityQueue,
)


RESNET_LAYERS = 2
TF_DIM = 128
TF_FC_DIM = 256
TF_DROPOUT = 0.4
TF_LAYERS = 2
TF_NHEAD = 4
MAX_OUTPUT_LEN = 250
BEAM_WIDTH = 5


def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    """Generate a triangular (size, size) mask."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class ResnetTransformer(BaseModel):
    """Resnet as encoder and transformer as decoder.

    Attributes:
        num_classes: Vocabulary size
        tf_dim: This serves multiple roles:
            - the output dimension of the encoder,
            - the input dimension of the decoder,
            - the dimension of feedforward networks in the transformer,
            - the dimension of label embeddings, and
            - the dimension of positional encoding.
        max_output_len: Maximum output length during inference.
        resnet: ResNet-18 model. Pretrained weights are not used because the
            input domain here is quite different from the original problem.
        encoder_projection: A convoluational layer with kernerl size of 1. It
            aims to reduce the number of channels.
        enc_pos_encoder: 2D positional encoding for the encoder.
        embedding: Embedding layer for the targets.
        y_mask: Mask to prevent attention to read tokens in future positions.
        dec_pos_encoder: 1D positional encoding for the decoder.
        transformer_decoder: Transformer decoder.
        fc: Fully connected layer. The output size must be num_classes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.resnet_layers = self.args.get("resnet-layers", RESNET_LAYERS)
        assert 0 <= self.resnet_layers <= 4
        self.tf_dim = self.args.get("tf-dim", TF_DIM)
        self.tf_fc_dim = self.args.get("tf-fc-dim", TF_FC_DIM)
        self.tf_nhead = self.args.get("tf-nhead", TF_NHEAD)
        self.tf_dropout = self.args.get("tf-dropout", TF_DROPOUT)
        self.tf_layers = self.args.get("tf-layers", TF_LAYERS)
        self.max_output_len = self.args.get("max-output-len", MAX_OUTPUT_LEN)
        self.beam_width = self.args.get("beam_width", BEAM_WIDTH)

        # Encoder
        resnet = torchvision.models.resnet18(pretrained=False)
        layers = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool]
        for i in range(1, self.resnet_layers + 1):
            layers.append(getattr(resnet, f"layer{i}"))
        self.resnet = nn.Sequential(*layers)
        # Get the output dimension of the last block of the last layer
        # layer1: 64, layer2: 128, layer3: 256, layer4: 512
        resnet_dim = layers[-1][-1].conv2.out_channels
        self.encoder_projection = nn.Conv2d(resnet_dim, self.tf_dim, 1)
        self.enc_pos_encoder = PositionalEncoding2D(self.tf_dim)

        # Decoder
        self.embedding = nn.Embedding(self.num_classes, self.tf_dim)
        self.y_mask = generate_square_subsequent_mask(self.max_output_len)
        self.dec_pos_encoder = PositionalEncoding1D(
            d_model=self.tf_dim, max_len=self.max_output_len
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                self.tf_dim, self.tf_nhead, self.tf_fc_dim, self.tf_dropout
            ),
            num_layers=self.tf_layers,
        )
        self.fc = nn.Linear(self.tf_dim, self.num_classes)

        # It is empirically important to initialize weights properly
        self.init_weights()

    def config(self) -> Dict[str, Any]:
        """Returns important configuration for reproducibility."""
        return {
            "resnet-layers": self.resnet_layers,
            "tf-dim": self.tf_dim,
            "tf-fc-dim": self.tf_fc_dim,
            "tf-nhead": self.tf_nhead,
            "tf-dropout": self.tf_dropout,
            "tf-layers": self.tf_layers,
            "max-output-len": self.max_output_len,
        }

    def init_weights(self) -> None:
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

        nn.init.kaiming_normal_(
            self.encoder_projection.weight.data,
            a=0,
            mode="fan_out",
            nonlinearity="relu",
        )
        if self.encoder_projection.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(
                self.encoder_projection.weight.data
            )
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(self.encoder_projection.bias, -bound, bound)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, _E, _H, _W)
            y: (B, Sy) with elements in (0, num_classes - 1)

        Returns:
            (B, num_classes, Sy) logits
        """
        encoded_x = self.encode(x)  # (Sx, B, E)
        output = self.decode(y, encoded_x)  # (Sy, B, num_classes)
        output = output.permute(1, 2, 0)  # (B, num_classes, Sy)
        return output

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs.

        Args:
            x: (B, _E, _H, _W)

        Returns:
            (Sx, B, E)
        """
        _E = x.shape[1]
        # Resnet expects 3 channels but training images are in gray scale
        if _E == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.resnet(x)  # (B, RESNET_DIM, H, W); H = _H // 32, W = _W // 32
        x = self.encoder_projection(x)  # (B, E, H, W)
        x = self.enc_pos_encoder(x)  # (B, E, H, W)
        x = torch.flatten(x, start_dim=2)  # (B, E, H * W)
        x = x.permute(2, 0, 1)  # (Sx, B, E); Sx = H * W
        return x

    def decode(self, y: torch.Tensor, encoded_x: torch.Tensor) -> torch.Tensor:
        """Decode encoded inputs with teacher-forcing.

        Args:
            encoded_x: (Sx, B, E)
            y: (B, Sy) with elements in (0, num_classes - 1)

        Returns:
            (Sy, B, num_classes) logits
        """
        y = y.permute(1, 0)  # (Sy, B)
        y = self.embedding(y) * math.sqrt(self.tf_dim)  # (Sy, B, E)
        y = self.dec_pos_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)  # (Sy, Sy)
        output = self.transformer_decoder(y, encoded_x, y_mask)  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, num_classes)
        return output

    def predict(
        self,
        x: torch.Tensor,
        beam_width: int = 1,
        max_output_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Make predctions at inference time.

        Args:
            x: (B, H, W) images
            beam_width: The number of sequences to store at each step. If
                smaller than or equal to 1, use greedy search.
            max_output_len: Maximum output length. Have to be smaller than or
                equal to the `max_len` in positional encoding.

        Returns:
            (B, max_output_len) with elements in (0, num_classes - 1).
        """
        if max_output_len is None:
            max_output_len = self.max_output_len
        elif max_output_len > self.max_output_len:
            raise ValueError(
                "max_output_len is expected to be smaller than "
                f"{self.max_output_len}"
            )

        if beam_width <= 1:
            output_indices = self._greedy_search(x, max_output_len)
        else:
            output_indices = self._beam_search(x, beam_width, max_output_len)
        return output_indices

    def _greedy_search(
        self,
        x: torch.Tensor,
        max_output_len: int,
    ) -> torch.Tensor:
        B = x.shape[0]
        S = max_output_len

        encoded_x = self.encode(x)  # (Sx, B, E)

        output_indices = (
            torch.full((B, S), self.pad_index).type_as(x).long()
        )  # (B, S)  # noqa: E501
        output_indices[:, 0] = self.sos_index
        for Sy in range(1, S):
            y = output_indices[:, :Sy]  # (B, Sy)
            logits = self.decode(y, encoded_x)  # (Sy, B, num_classes)
            output = torch.argmax(logits, dim=-1)  # (Sy, B)
            output_indices[:, Sy] = output[-1:]  # Set the last output token

            # Early stopping of prediction loop to speed up prediction
            current_indices = output_indices[:, Sy]
            is_ended = current_indices == self.eos_index
            is_padded = current_indices == self.pad_index
            if (is_ended | is_padded).all():
                break

        # Set all tokens after end token to be padding
        for Sy in range(1, S):
            previous_indices = output_indices[:, Sy - 1]
            is_ended = previous_indices == self.eos_index
            is_padded = previous_indices == self.pad_index
            output_indices[(is_ended | is_padded), Sy] = self.pad_index

        return output_indices

    def _beam_search(
        self,
        x: torch.Tensor,
        beam_width: int,
        max_output_len: int,
    ) -> torch.Tensor:
        B = x.shape[0]
        S = max_output_len
        k = beam_width

        encoded_x = self.encode(x)  # (Sx, B, E)
        output_indices = torch.full((B, S), self.pad_index).type_as(x).long()

        # Loop over each sample in the batch
        for i in range(B):
            initial_seq = torch.full((S,), self.pad_index).type_as(x).long()
            initial_seq[0] = self.sos_index
            initial_candidate = BeamSearchCandidate(
                log_likelihood=0,
                seq=initial_seq,
                current_seq_len=1,
                eos_index=self.eos_index,
            )
            queue = TopKPriorityQueue(k)
            queue.push(initial_candidate)

            while True:
                # Create a fixed-size priority queue (min heap). Only keep the
                # top k candidates to save memory.
                new_queue = TopKPriorityQueue(k)
                for candidate in queue:
                    if candidate.has_ended():
                        new_queue.push(candidate)
                        continue
                    y = candidate.seq[: len(candidate)].unsqueeze(0)  # (1, Sy)
                    logits = self.decode(
                        y, encoded_x[:, i, :]
                    )  # (Sy, 1, num_classes)
                    logits = logits.squeeze(1)  # (Sy, num_classes)
                    log_probs = torch.log_softmax(
                        logits, dim=1
                    )  # (Sy, num_classes)
                    top_k_log_probs, top_k_indices = log_probs.topk(k)
                    for log_prob, index in zip(top_k_log_probs, top_k_indices):
                        new_candidate = candidate.extend(log_prob, index)
                        new_queue.push(new_candidate)
                queue = new_queue
                # Stop the search if all candidates have already generated the
                # end-of-sequence token
                all_ended = all(candidate.has_ended() for candidate in queue)
                if all_ended:
                    break

            best_candidate = queue.get_largest_item(keep_items=False)
            output_indices[i, :] = best_candidate.seq

        return output_indices
