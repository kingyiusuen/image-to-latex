from image_to_latex.utils.metrics import edit_distance


class TestEditDistance:
    def test_zero_match(self):
        references = [[1, 2, 3], [4, 5, 6]]
        hypotheses = [[4, 5, 6], [1, 2, 3]]
        assert edit_distance(references, hypotheses) == 0

    def test_perfect_match(self):
        references = [[1, 2, 3], [4, 5, 6]]
        hypotheses = [[1, 2, 3], [4, 5, 6]]
        assert edit_distance(references, hypotheses) == 1

    def test_half_match(self):
        references = [[1, 2, 3], [4, 5, 6]]
        hypotheses = [[1, 2, 3], []]
        assert edit_distance(references, hypotheses) == 0.5

    def test_different_lengths(self):
        references = [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10]]
        hypotheses = [[1, 2, 3, 4, 5, 6, 7], []]
        assert edit_distance(references, hypotheses) == 0.7
