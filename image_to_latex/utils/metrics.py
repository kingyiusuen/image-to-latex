# Reference: https://github.com/luopeixiang/im2latex/blob/master/model/score.py

import editdistance
import nltk


def bleu_score(references, hypotheses) -> float:
    """Computes bleu score.

    Args:
        references: List of lists of tokens
        hypotheses: List of lists of tokens

    Returns:
        BLEU-4 score: higher is better, 1 is perfect.
    """
    references = [[ref] for ref in references]  # for corpus_bleu func
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(
        references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)
    )
    return BLEU_4


def edit_distance(references, hypotheses) -> float:
    """Computes Levenshtein distance between two sequences.

    Args:
        references: List of lists of tokens
        hypotheses: List of lists of tokens

    Returns:
        1 - levenshtein distance: higher is better, 1 is perfect.
    """
    d_leven, len_tot = 0., 0.
    for ref, hypo in zip(references, hypotheses):
        d_leven += editdistance.distance(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))
    return 1.0 - d_leven / len_tot
