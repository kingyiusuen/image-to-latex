from typing import Union

import numpy as np
import torch
from torch_lr_finder import LRFinder

from image_to_latex.models import ResnetTransformer
from image_to_latex.utils.misc import find_first


class LRFinder_(LRFinder):
    """Learning rate range test.

    `LRFinder_` inherits `LRFinder` of the package `torch_lr_finder`. Some
    methods are modified to fit our use cases.

    References:
    https://github.com/davidtvs/pytorch-lr-finder
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _train_batch(self, train_iter, accumulation_steps, non_blocking_transfer=True):
        self.model.train()
        total_loss = None  # for late initialization
        self.optimizer.zero_grad()
        for _ in range(accumulation_steps):
            imgs, targets = next(train_iter)
            imgs, targets = self._move_to_device(
                imgs, targets, non_blocking=non_blocking_transfer
            )
            # Forward pass
            if isinstance(self.model, ResnetTransformer):
                outputs = self.model(imgs, targets[:, :-1])
                loss = self.criterion(outputs, targets[:, 1:])
            else:
                logprobs = self.model(imgs)  # (B, num_classes, S)
                B, _, S = logprobs.shape
                logprobs_for_loss = logprobs.permute(2, 0, 1)  # (S, B, num_classes)
                input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S
                target_lengths = find_first(targets, element=self.model.padding_index)
                target_lengths = target_lengths.type_as(targets)
                loss = self.criterion(logprobs_for_loss, targets, input_lengths, target_lengths)

            # Loss should be averaged in each step
            loss /= accumulation_steps
            # Backward pass
            loss.backward()
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
        self.optimizer.step()
        return total_loss.item()

    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> Union[float, None]:
        """Plots the learning rate range test.

        Args:
            skip_start: number of batches to trim from the start.
            skip_end: number of batches to trim from the start.

        Returns:
            The suggested learning rate.

        Raises:
            ValueError: If fails to compute the gradients.
        """
        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary. Also, handle
        # skip_end=0 properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Suggest the point with steepest gradient (minimal gradient)
        min_grad_idx = None
        try:
            min_grad_idx = (np.gradient(np.array(losses))).argmin()
        except ValueError:
            print("Failed to compute the gradients, there might not be enough points.")
        if min_grad_idx is not None:
            print("Suggested LR: {:.2E}".format(lrs[min_grad_idx]))
            return lrs[min_grad_idx]
        return None
