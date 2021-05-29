import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import torch
from PIL import Image


class Tokenizer:
    """
    Args:
        token_to_index: A dictionary that maps tokens to indices.

    Attributes:
        blk_token: Blank token for CTC loss.
        sos_token: Start-of-sequence token.
        eos_token: End-of-sequence token.
        pad_token: Padding token.
        oov_token: Out-of-vocabulary token.
        special_tokens: A list of special tokens (blank token,
            start-of-sequence token, end-of-sequence token, padding token, and
            out-of-vocabulary token).
        blk_index: Index of blank token.
        sos_index: Index of start-of-sequence token.
        eos_index: Index of end-of-sequence token.
        pad_index: Index of padding token.
        oov_index: Index of out-of-vocabulary token.
        token_to_index: A dictionary that maps tokens to indices.
        index_to_token: A dictionary that maps indices to tokens.
    """

    def __init__(
        self,
        token_to_index: Optional[Dict[str, int]] = None,
    ) -> None:
        self.token_to_index: Dict[str, int]
        self.index_to_token: Dict[int, str]

        self.blk_token = "<BLK>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.oov_token = "<UNK>"
        self.special_tokens = ["<BLK>", "<SOS>", "<EOS>", "<PAD>", "<UNK>"]
        if token_to_index:
            assert "<BLK>" in token_to_index
            assert "<SOS>" in token_to_index
            assert "<EOS>" in token_to_index
            assert "<PAD>" in token_to_index
            assert "<UNK>" in token_to_index
            self.token_to_index = token_to_index
        else:
            self.token_to_index = {
                self.blk_token: 0,
                self.sos_token: 1,
                self.eos_token: 2,
                self.pad_token: 3,
                self.oov_token: 4,
            }
        self.blk_index = self.token_to_index[self.blk_token]
        self.sos_index = self.token_to_index[self.sos_token]
        self.eos_index = self.token_to_index[self.eos_token]
        self.pad_index = self.token_to_index[self.pad_token]
        self.oov_index = self.token_to_index[self.oov_token]
        self.index_to_token = {
            index: token for token, index in self.token_to_index.items()
        }

    def __len__(self):
        return len(self.token_to_index)

    def build(
        self,
        corpus: Iterable[Iterable[str]],
        min_count: int = 2,
    ) -> None:
        """Create a mapping from tokens to indices and vice versa.

        Args:
            texts: List of texts made of tokens.
            min_count: Tokens that appear fewer than `min_count` will not be
                included in the mapping.
        """
        # Count the frequency of each token
        counter: Dict[str, int] = {}
        for sentence in corpus:
            for token in sentence:
                counter[token] = counter.get(token, 0) + 1

        for token, count in counter.items():
            # Remove tokens that show up fewer than `min_count` times
            if count < min_count:
                continue
            index = len(self)
            self.index_to_token[index] = token
            self.token_to_index[token] = index

    def index(
        self,
        corpus: Iterable[Iterable[str]],
        add_sos: bool = False,
        add_eos: bool = False,
        pad_to: Optional[int] = None,
    ) -> List[List[int]]:
        """Convert a corpus of tokens to indices.

        Args:
            corpus: Corpus that is being indexed.
            add_sos: Whether to add a start-of-sequence token at the
                beginning of each sequence.
            add_eos: Whether to add a end-of-sequence token at the end of
                each sequence.
            pad_to: Indicates the length all sequences should be padded to.
        """
        indexed_corpus = []
        for sentence in corpus:
            indexed_sentence = []
            for token in sentence:
                index = self.token_to_index.get(token, self.oov_index)
                indexed_sentence.append(index)
            if add_sos:
                indexed_sentence = [self.sos_index] + indexed_sentence
            if add_eos:
                indexed_sentence += [self.eos_index]
            if pad_to is not None:
                pad_len = pad_to - len(indexed_sentence)
                if pad_len < 0:
                    raise RuntimeError(
                        f"Sentence '{sentence}' is longer than specified "
                        f"padded length (specified {pad_to}, found "
                        f"{len(indexed_sentence)})."
                    )
                else:
                    indexed_sentence += [self.pad_index] * pad_len
            indexed_corpus.append(indexed_sentence)
        return indexed_corpus

    def unindex(
        self,
        indexed_corpus: Iterable[Iterable[int]],
        inference: bool = False,
    ) -> List[List[str]]:
        """Convert corpus of indices to original tokens.

        Args:
            indexed_corpus: Indexed corpus that is being unindexed.
            inference: If True, break after the first end-of-sequence token,
                and ignore special tokens.
        """
        if isinstance(indexed_corpus, torch.Tensor):
            indexed_corpus = indexed_corpus.tolist()
        corpus = []
        for indexed_sentence in indexed_corpus:
            sentence = []
            for index in indexed_sentence:
                token = self.index_to_token[index]
                if inference and token == self.eos_token:
                    break
                if inference and token in self.special_tokens:
                    continue
                sentence.append(token)
            corpus.append(sentence)
        return corpus

    def save(self, filename: Union[Path, str]):
        """Save token-to-index mapping to a json file."""
        with open(filename, "w") as f:
            json.dump(self.token_to_index, f, indent=4)


def resize_image(image: Image, width: int, height: int) -> Image:
    """Resize an image while keeping its aspect ratio in a white background.

    Reference:
    https://stackoverflow.com/a/52969463
    """
    ratio_w = width / image.width
    ratio_h = height / image.height
    if ratio_w < ratio_h:
        resized_width = width
        resized_height = round(ratio_w * image.height)
    else:
        resized_width = round(ratio_h * image.width)
        resized_height = height
    resized_image = image.resize(
        (resized_width, resized_height),
        resample=Image.ANTIALIAS,
    )
    # Create a white background
    background = Image.new("L", (width, height), 255)
    # Paste in the center of the background
    offset = (
        round((width - resized_width) / 2),
        round((height - resized_height) / 2),
    )
    background.paste(resized_image, offset)
    return background
