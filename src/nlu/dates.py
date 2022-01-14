""" Value extraction for dates

Copyright PolyAI Limited
"""
import datetime as dt
from typing import List, Optional

import dateparser
from dateparser.search import search_dates
from dateparser_data.settings import default_parsers

from nlu.interface import AbstractValueExtractor


def date_to_str(dob: Optional[dt.date]) -> str:
    """ Convert a date object to string """
    return dob.isoformat() if dob is not None else ''


class DummyDateValueExtractor(AbstractValueExtractor):
    """ Dummy date NLU that always returns a dummy date """

    def extract(self, text: str) -> List[dt.date]:  # noqa D003
        return [dt.date(2000, 1, 1)]


class EviDateValueExtractor(AbstractValueExtractor):
    """ Simple date NLU """

    def __init__(
        self,
        locale: str,
        strict: bool,
        use_nbest: bool
    ):
        """ Initialise

        Args:
            locale: the language-region locale
            strict: whether extract exact matches only
            use_nbest: whether extract from the ASR n-best
        """
        super().__init__(
            locale=locale,
            strict=strict,
            use_nbest=use_nbest
        )
        #
        # see https://dateparser.readthedocs.io/en/latest/settings.html
        parsers = [
            parser for parser in default_parsers
            if parser != 'relative-time'
        ]
        if not self._strict:
            parsers += ['no-spaces-time']
        self._settings = {
            'PARSERS': parsers,
            'STRICT_PARSING': True,
            'PREFER_DATES_FROM': 'past',
            'RELATIVE_BASE': dt.datetime(2020, 1, 1)
        }

    def extract(  # noqa D003
        self,
        text: str,
        flags: Optional[List[str]] = None
    ) -> List[dt.date]:
        dates = []
        #
        # exact match
        d = dateparser.parse(
            text,
            languages=[self.lang],
            settings=self._settings
        )
        if isinstance(d, dt.datetime):
            dates.append(d.date())
        #
        # search within context
        if not self._strict:
            for d in search_dates(
                text, languages=[self.lang], settings=self._settings
            ) or []:
                dates.append(d[1].date())
        #
        return dates
