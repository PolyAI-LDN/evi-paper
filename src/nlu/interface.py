""" NLU abstract class

Copyright PolyAI Limited
"""
from __future__ import annotations

import abc
from typing import Any, List, Optional


class AbstractValueExtractor(abc.ABC):
    """ Abstract class for NLU extractors """

    def __init__(
        self,
        locale: str,
        strict: bool = False,
        use_nbest: bool = False
    ):
        """ Initialise

        Args:
            locale: the language-region locale
            strict: whether extract exact matches only
            use_nbest: whether extract from the ASR n-best
        """
        self._locale = locale
        self._strict = strict
        self._use_nbest = use_nbest

    @property
    def lang(self):
        """Language of the NLU model"""
        return self._locale.split('-')[0].lower()

    @abc.abstractmethod
    def extract(
        self,
        text: str,
        flags: Optional[List[str]] = None
    ) -> List[Any]:
        """ Parse text into a date object

        Args:
            text: the input text to process
            flags: a list of special flags for optional processing
        """
        raise NotImplementedError()

    def extract_nbest(
        self,
        texts: List[str],
        flags: Optional[List[str]] = None
    ) -> List[Any]:
        """ Parse a list of texts into dates

        Args:
            texts: a list of input texts to process
            flags: a list of special flags for optional processing

        Returns:
            a list of extracted values
        """
        results = []
        for t in (texts if self._use_nbest else texts[:1]):
            results.extend(self.extract(text=t, flags=flags))
        results = _rm_duplicates(results)
        return results


def _rm_duplicates(alist: List[Any]) -> List[Any]:
    """ Remove duplicates from a list

    Args:
        alist: the input list

    Returns:
        a copy of the list with any duplicates removed
    """
    aset = set()
    res = []
    for a in alist:
        if a in aset:
            continue
        res.append(a)
        aset.add(a)
    return res
