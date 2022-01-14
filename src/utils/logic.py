""" Models for verification scoring

Copyright PolyAI Limited
"""

from functools import partial
from typing import Iterable, List, Optional, Tuple

import numpy as np


class Logic(object):
    """ Infinity-one p-norm logic to combine item scores to profile scores """

    def __init__(self, alpha: float = 1):
        """ Initialise

        Args:
            alpha: the interpolation value of the infinity-one norm
        """
        assert 0 <= alpha <= 1
        if alpha == 1:
            self._and_op = zadeh_and
            self._or_op = zadeh_or
        else:
            self._and_op = partial(p_norm_and, alpha=alpha)
            self._or_op = partial(p_norm_or, alpha=alpha)

    def combine_scores(
        self,
        score_postcode: float,
        score_name_full: float,
        scores_name_parts: List[Tuple[float, float]],
        score_dob: float,
    ) -> Optional[float]:
        """ The operator that combines the item scores to profile score

        Args:
            score_postcode: the verification score for postcode matching
            score_name_full: the verification score for full name matching
            scores_name_parts: the verification scores for part name matching
            score_dob: the verification score for date matching

        Returns:
            the profile-level score
        """
        # print(score_postcode, score_name_full, score_dob)
        return self._and_op([
            score_postcode,
            self._or_op([
                score_name_full,
                max([
                    self._and_op([score_first, score_last])
                    for score_first, score_last in scores_name_parts
                ] + [0.0]),
            ]),
            score_dob,
        ])


def zadeh_and(values: Iterable[float]) -> float:
    """ Zadeh's AND operator """
    return min(values)


def zadeh_or(values: Iterable[float]) -> float:
    """ Zadeh's OR operator """
    return max(values)


def p_norm_and(values: Iterable[float], alpha: float) -> float:
    """ p-norm's AND operator (as infinity-one norm)

    Args:
        values: the arguments of the operator
        alpha: the interpolation value

    Returns:
        the result of the application of the operator to the arguments
    """
    assert (0 <= alpha <= 1)
    values = list(values)
    return alpha * min(values) + (1 - alpha) * np.mean(values)


def p_norm_or(values: Iterable[float], alpha: float) -> float:
    """ p-norm's OR operator (as infinity-one norm)

    Args:
        values: the arguments of the operator
        alpha: the interpolation value

    Returns:
        the result of the application of the operator to the arguments
    """
    assert (0 <= alpha <= 1)
    values = list(values)
    return alpha * max(values) + (1 - alpha) * np.mean(values)
