from typing import Dict, List, Sequence

import torch


def convert_strings_to_labels(
    strings: Sequence[Sequence[str]], token2id: Dict[str, int], seq_length: int
) -> torch.Tensor:
    """Convert strings to labels in the form of a fixed-length tensor.

    Convert N strings (which has already been split into tokens) to a
    (N, seq_length) ndarray, with each string wrapped with begin of sequence
    and end of sequence tokens, and padded with the padding token.
    """
    unknown_index = token2id["<UNK>"]
    labels = torch.ones((len(strings), seq_length), dtype=torch.long)
    labels *= token2id["<PAD>"]
    for i, formula in enumerate(strings):
        tokens = list(formula)
        tokens = ["<BOS>", *tokens, "<EOS>"]
        for j, token in enumerate(tokens):
            labels[i, j] = float(token2id.get(token, unknown_index))
    return labels


def convert_labels_to_strings(
    labels: torch.Tensor, id2token: List[str], ignored_tokens: Sequence[str] = []
) -> List[List[str]]:
    """Convert labels to strings.

    The start token at the beginning will be removed. Any tokens after (and
    including) the first end token will be removed. The output strings may have
    different lengths.
    """
    # Remove start token
    labels = labels[:, 1:]

    # Remove tokens after the end token
    strings = []
    for label in labels.tolist():
        string = []
        for index in label:
            token = id2token[index]
            if token == "<EOS>":
                break
            if token in ignored_tokens:
                continue
            string.append(token)
        strings.append(string)
    return strings
