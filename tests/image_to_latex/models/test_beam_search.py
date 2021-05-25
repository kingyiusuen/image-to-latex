import pytest
import torch

from image_to_latex.models.beam_search import (
    BeamSearchCandidate,
    TopKPriorityQueue,
)


class TestBeamSearchCandidate:
    @pytest.fixture
    def candidate1(self):
        return BeamSearchCandidate(
            log_likelihood=0.2,
            seq=torch.Tensor([1, 3, 4, 2, 0, 0, 0]),
            current_seq_len=4,
            eos_index=6,
        )

    @pytest.fixture
    def candidate2(self):
        return BeamSearchCandidate(
            log_likelihood=0.7,
            seq=torch.Tensor([1, 3, 4, 2, 6, 0, 0]),
            current_seq_len=5,
            eos_index=6,
        )

    def test_len(self, candidate1):
        assert len(candidate1) == 4

    def test_score(self, candidate1):
        assert abs(candidate1.score() - 0.066666) < 1e-6

    def test_extend(self, candidate1, candidate2):
        new_candidate = candidate1.extend(0.5, 6)
        assert new_candidate.log_likelihood == candidate2.log_likelihood
        assert torch.equal(new_candidate.seq, candidate2.seq)
        assert new_candidate.current_seq_len == candidate2.current_seq_len
        assert new_candidate.eos_index == candidate2.eos_index

    def test_has_ended(self, candidate1, candidate2):
        assert not candidate1.has_ended()
        assert candidate2.has_ended()

    def test_comparators(self, candidate1, candidate2):
        assert candidate1 != candidate2
        assert candidate1 < candidate2
        assert candidate1 <= candidate2
        assert candidate2 > candidate1
        assert candidate2 >= candidate1
        assert candidate1.extend(0.5, 6) == candidate2


class TestTopKPriorityQueue:
    @pytest.fixture
    def queue(self):
        pq = TopKPriorityQueue(capacity=3)
        pq.push(5)
        pq.push(-3)
        pq.push(1)
        return pq

    def test_push(self, queue):
        assert queue[0] == -3
        assert len(queue) == 3
        queue.push(2)
        assert queue[0] == 1
        assert len(queue) == 3

    def test_pop(self, queue):
        assert queue.pop() == -3

    def test_get_largest_item(self, queue):
        assert queue.get_largest_item() == 5
