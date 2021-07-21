from typing import Set

import editdistance
import torch
from torch import Tensor
from torchmetrics import Metric


class EditDistance(Metric):
    def __init__(self, ignore_tokens: Set[int], *args):
        super().__init__(*args)
        self.ignore_tokens = ignore_tokens
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.error: Tensor
        self.total: Tensor

    def update(self, preds, targets):
        N = preds.shape[0]
        for i in range(N):
            pred = [token for token in preds[i].tolist() if token not in self.ignore_tokens]
            target = [token for token in targets[i].tolist() if token not in self.ignore_tokens]
            distance = editdistance.distance(pred, target)
            self.error += distance / max(len(pred), len(target))
        self.total += N

    def compute(self) -> Tensor:
        return 1 - self.error / self.total


class ExactMatch(Metric):
    def __init__(self, ignore_tokens: Set[int], *args):
        super().__init__(*args)
        self.ignore_tokens = ignore_tokens
        self.add_state("matches", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.matches: Tensor
        self.total: Tensor

    def update(self, preds, targets):
        N = preds.shape[0]
        for i in range(N):
            pred = [token for token in preds[i].tolist() if token not in self.ignore_tokens]
            target = [token for token in targets[i].tolist() if token not in self.ignore_tokens]
            self.matches += 1 if pred == target else 0
        self.total += N

    def compute(self) -> Tensor:
        return self.matches / self.total
