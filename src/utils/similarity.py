""" Models for verification scoring

Copyright PolyAI Limited
"""

import abc
import datetime as dt
from hashlib import md5
from typing import Any, List

from Levenshtein import distance as edit_distance
from numpy.random import RandomState


class BaseScorer(abc.ABC):
    """Calculator for verification match scores"""

    @abc.abstractmethod
    def calc_score(self, hyps: List[Any], ref: Any) -> float:
        """Calculate a match score between list of hypotheses and reference

        Args:
            hyps: a list of hypotheses
            ref: the reference value to compare against
        """
        raise NotImplementedError()


class FallbackScorer(BaseScorer):
    """ A scorer that sequentially examines all hypotheses """

    def _preprocess(self, ref: Any) -> Any:
        if isinstance(ref, str):
            ref = ref.strip().replace(" ", "").replace("-", "").upper()
        elif isinstance(ref, dt.date):
            ref = ref.isoformat()
            ref = ref.replace(" ", "").replace(":", "").replace("-", "")
        else:
            raise NotImplementedError(f"Not supported type {type(ref)}")
        return ref

    def calc_score(  # noqa D003
        self,
        hyps: List[Any],
        ref: Any
    ) -> float:
        max_s = 0.0
        _ref = self._preprocess(ref)
        for hyp in hyps:
            if hyp is None:
                continue
            _hyp = self._preprocess(hyp)
            s = self.calc_score_single(_hyp, _ref)
            # rolling max over all items in nbest
            max_s = max(s, max_s)
            # early crisp stop
            if max_s == 1.0:
                return 1.0
        return max_s

    @abc.abstractmethod
    def calc_score_single(self, hyp: Any, ref: Any) -> float:
        """Calculate a match score between single hypotheses and reference

        Args:
            hyp: a single hypothesis
            ref: the reference value to compare against
        """
        raise NotImplementedError()


class AlwaysTrueScorer(FallbackScorer):
    """ A scorer that always returns True """

    def calc_score_single(  # noqa D003
        self, hyp: Any, ref: Any
    ) -> float:
        return 1.0


class ConstantScorer(FallbackScorer):
    """ A scorer that outputs a constant score """

    def __init__(self, const: float = 0.5):
        """ Initialise

        Args:
            const: the value of the constant score output
        """
        assert 0 <= const <= 1
        self._const = const

    def calc_score(  # noqa D003
        self,
        hyps: List[Any],
        ref: Any
    ) -> float:
        return self._const

    def calc_score_single(  # noqa D003
        self, hyp: Any, ref: Any
    ) -> float:
        return self._const


class ExactScorer(FallbackScorer):
    """ A scorer that checks for equals (==) """

    def calc_score_single(  # noqa D003
        self, hyp: Any, ref: Any
    ) -> float:
        return float(hyp == ref)


class TextEditScorer(FallbackScorer):
    """ A scorer that checks performs edit distance comparison """

    def calc_score_single(  # noqa D003
        self, hyp: Any, ref: Any,
    ) -> float:
        if ref == hyp:
            return 1.0
        if not ref or not hyp:
            return 0.0
        nom = edit_distance(ref, hyp)
        denom = max(len(ref), len(hyp))
        sim = 1 - nom / denom
        return sim


class RandomScorer(FallbackScorer):
    """ A scorer that checks performs random comparison """

    def calc_score_single(  # noqa D003
        self, hyp: Any, ref: Any,
    ) -> float:
        if ref == hyp:
            return 1.0
        seed = int(
            md5(
                '|'.join([str(ref), str(hyp)]).encode("utf-8")
            ).hexdigest()[:6],
            16
        )
        rng = RandomState(seed=seed)
        sim = rng.uniform(0, 1)
        return sim
