from __future__ import annotations

import heapq
from typing import Any

import torch


class BeamSearchCandidate:
    """A class for storing a candidate sequence in beam search.

    Args:
        log_likelihood: Unnormalized log likelihood of the sequence.
        seq: (max_seq_len) with elements in (0, num_classes - 1).
        seq_len: Length of the sequence, not counting the padding token.
        eos_index: Index of end-of-sequence token.
    """

    def __init__(
        self,
        log_likelihood: float,
        seq: torch.Tensor,
        max_seq_len: int,
        eos_index: int,
    ) -> None:
        self.log_likelihood = log_likelihood
        self.seq = seq
        self.max_seq_len = max_seq_len
        self.eos_index = eos_index

    def __len__(self) -> int:
        return len(self.seq)

    def has_ended(self) -> bool:
        eos_generated = self.seq[-1].item() == self.eos_index
        max_len_reached = len(self) == self.max_seq_len
        return eos_generated | max_len_reached

    def extend(self, log_prob: float, index: int) -> BeamSearchCandidate:
        if len(self) == self.max_seq_len:
            return self
        return BeamSearchCandidate(
            self.log_likelihood + log_prob,
            torch.cat((self.seq, torch.LongTensor([index]))),
            self.max_seq_len,
            self.eos_index,
        )

    def score(self) -> float:
        return self.log_likelihood / float(len(self) - 1 + 1e-6)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, BeamSearchCandidate):
            raise NotImplementedError
        return self.score() < other.score()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, BeamSearchCandidate):
            raise NotImplementedError
        return self.score() <= other.score()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BeamSearchCandidate):
            raise NotImplementedError
        return self.score() == other.score()


class TopKPriorityQueue:
    """Priority queue implemented using a min heap.

    The queue will only store the k-th largest items that we have tried to push
    to it.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.queue: list[Any] = []

    def __len__(self) -> int:
        return len(self.queue)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self):
            item = self.queue[self.i]
            self.i += 1
            return item
        else:
            raise StopIteration

    def __getitem__(self, i: int) -> Any:
        return self.queue[i]

    def push(self, item: Any) -> None:
        """Add item if it is larger than the k-th largest in the queue."""
        if len(self.queue) < self.capacity:
            heapq.heappush(self.queue, item)
        elif item > self.queue[0]:
            heapq.heappushpop(self.queue, item)

    def pop(self) -> Any:
        """Returns the smallest item."""
        return heapq.heappop(self.queue)

    def get_largest_item(self, keep_items: bool = True) -> Any:
        """Get the largest item in the queue.

        Args:
            keep_items: All the items will have to be popped out to find the
                largest item, as it is at the bottom of the queue. If
                `keep_items` is True, all the items will be put back into the
                queue.
        """
        if keep_items:
            tmp_queue = self.queue.copy()
        while self.queue:
            item = self.pop()
        if keep_items:
            self.queue = tmp_queue
        return item
