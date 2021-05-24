import pytest

from image_to_latex.utils.data import Tokenizer


class TestTokenizer:
    @pytest.fixture
    def token_to_index(self):
        return {
            "<BLK>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<PAD>": 3,
            "<UNK>": 4,
            "a": 5,
            "b": 6,
        }

    @pytest.fixture
    def index_to_token(self):
        return {
            0: "<BLK>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<PAD>",
            4: "<UNK>",
            5: "a",
            6: "b",
        }

    def test_init(self, token_to_index, index_to_token):
        tokenizer = Tokenizer(token_to_index)
        assert tokenizer.token_to_index == token_to_index
        assert tokenizer.index_to_token == index_to_token
        assert len(tokenizer) == 7

    def test_build(self, token_to_index, index_to_token):
        tokenizer = Tokenizer()
        corpus = [["a", "b"], ["a", "b", "a"]]
        tokenizer.build(corpus)
        assert tokenizer.token_to_index == token_to_index
        assert tokenizer.index_to_token == index_to_token
        assert len(tokenizer) == 7

    def test_index_non_special_tokens(self, token_to_index):
        tokenizer = Tokenizer(token_to_index)
        corpus = [["a", "b"], ["a", "b", "a"]]
        indexed_corpus = tokenizer.index(corpus)
        assert indexed_corpus == [[5, 6], [5, 6, 5]]

    def test_index_add_sos(self, token_to_index):
        tokenizer = Tokenizer(token_to_index)
        corpus = [["a", "b"], ["a", "b", "a"]]
        indexed_corpus = tokenizer.index(corpus, add_sos=True)
        assert indexed_corpus == [[1, 5, 6], [1, 5, 6, 5]]

    def test_index_add_eos(self, token_to_index):
        tokenizer = Tokenizer(token_to_index)
        corpus = [["a", "b"], ["a", "b", "a"]]
        indexed_corpus = tokenizer.index(corpus, add_eos=True)
        assert indexed_corpus == [[5, 6, 2], [5, 6, 5, 2]]

    def test_index_add_sos_eos(self, token_to_index):
        tokenizer = Tokenizer(token_to_index)
        corpus = [["a", "b"], ["a", "b", "a"]]
        indexed_corpus = tokenizer.index(corpus, add_sos=True, add_eos=True)
        assert indexed_corpus == [[1, 5, 6, 2], [1, 5, 6, 5, 2]]

    def test_index_pad_to(self, token_to_index):
        tokenizer = Tokenizer(token_to_index)
        corpus = [["a", "b"], ["a", "b", "a"]]
        indexed_corpus = tokenizer.index(corpus, pad_to=3)
        assert indexed_corpus == [[5, 6, 3], [5, 6, 5]]

    def test_index_longer_than_padded_len(self, token_to_index):
        tokenizer = Tokenizer(token_to_index)
        corpus = [["a", "b"], ["a", "b", "a"]]
        with pytest.raises(Exception):
            tokenizer.index(corpus, pad_to=2)

    def test_index_add_sos_eos_pad_to(self, token_to_index):
        tokenizer = Tokenizer(token_to_index)
        corpus = [["a", "b"], ["a", "b", "a"]]
        indexed_corpus = tokenizer.index(
            corpus, add_sos=True, add_eos=True, pad_to=5
        )
        assert indexed_corpus == [[1, 5, 6, 2, 3], [1, 5, 6, 5, 2]]

    def test_index_unknown_token(self, token_to_index):
        tokenizer = Tokenizer(token_to_index)
        corpus = [["a", "c"], ["c", "b"]]
        indexed_corpus = tokenizer.index(corpus)
        assert indexed_corpus == [[5, 4], [4, 6]]

    def test_unindex_non_special_tokens(self, token_to_index):
        tokenizer = Tokenizer(token_to_index)
        indexed_corpus = [[5], [5, 6]]
        corpus = tokenizer.unindex(indexed_corpus)
        assert corpus == [["a"], ["a", "b"]]

    def test_unindex_inference(self, token_to_index):
        tokenizer = Tokenizer(token_to_index)
        indexed_corpus = [[1, 5, 2, 3], [1, 4, 5, 6, 2]]
        corpus = tokenizer.unindex(indexed_corpus, inference=True)
        assert corpus == [["a"], ["a", "b"]]
