""" Value extraction for postcodes

Copyright PolyAI Limited
"""
import re
from typing import List, Optional

from nlu.interface import AbstractValueExtractor
from utils.spelling import normalise_spellings


def _get_postcode_regex(locale: str):
    if locale == 'en-GB':
        return r'([Gg][Ii][Rr] 0[Aa]{2})|((([A-Za-z][0-9]{1,2})|(([A-Za-z][A-Ha-hJ-Yj-y][0-9]{1,2})|(([A-Za-z][0-9][A-Za-z])|([A-Za-z][A-Ha-hJ-Yj-y][0-9][A-Za-z]?))))\s?[0-9][A-Za-z]{2})'  # noqa
    elif locale == 'en-US':
        return r'(^\d{5}$)|(^\d{9}$)|(^\d{5}-\d{4}$)'
    elif locale == 'pl-PL':
        return r'\d{2}-?\d{3}'
    elif locale == 'fr-FR':
        return r'\d{5}'
    else:
        raise NotImplementedError(f'No postcode regex for {locale}')


def _standardise_postcode(text: str) -> str:
    return text.replace(" ", "").replace('-', '').upper()


def _validate_postcode(postcode: str, locale: str) -> str:
    """ Format a postcode to valid format (or empty string if invalid)"""
    postcode = _standardise_postcode(postcode)
    pattern = _get_postcode_regex(locale)
    if re.fullmatch(pattern, postcode):
        return postcode
    return ''


class EviPostcodeValueExtractor(AbstractValueExtractor):
    """ Postcode NLU """

    def extract(  # noqa D003
        self,
        text: str,
        flags: Optional[List[str]] = None
    ) -> List[str]:
        postcodes = []
        text = normalise_spellings(text, locale=self._locale)
        pattern = _get_postcode_regex(locale=self._locale)
        #
        # exact match
        p = _validate_postcode(
            _standardise_postcode(text),
            locale=self._locale
        )
        if p:
            postcodes.append(p)
        #
        if not self._strict:
            # search within context (may overlap)
            for match in re.finditer(rf'(?=({pattern}))', text):
                p = match.group(1)
                p = _standardise_postcode(p)
                postcodes.append(p)
        return postcodes
