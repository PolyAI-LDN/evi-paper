""" Models for NLU

Copyright PolyAI Limited
"""
from __future__ import annotations

from nlu.nlu import Nlu
from nlu.dates import EviDateValueExtractor
from nlu.names import EviNameValueExtractor
from nlu.postcodes import EviPostcodeValueExtractor


def build_nlu(name: str, locale: str) -> Nlu:
    """ Build NLU model for the verification task

    Args:
        name: the name of the model
        locale: the locale of the model, e.g. en-US, en-GB

    Returns:
        an NLU model
    """
    locale = locale.replace("_", "-")
    if name == 'cautious':
        return Nlu(
            postcode_extractor=EviPostcodeValueExtractor(
                locale=locale,
                strict=True,
                use_nbest=True,
            ),
            name_extractor=EviNameValueExtractor(
                locale=locale,
                strict=True,
                use_nbest=True,
            ),
            date_extractor=EviDateValueExtractor(
                locale=locale,
                strict=True,
                use_nbest=True,
            )
        )
    elif name == 'seeking':
        return Nlu(
            postcode_extractor=EviPostcodeValueExtractor(
                locale=locale,
                strict=False,
                use_nbest=True,
            ),
            name_extractor=EviNameValueExtractor(
                locale=locale,
                strict=False,
                use_nbest=True,
            ),
            date_extractor=EviDateValueExtractor(
                locale=locale,
                strict=False,
                use_nbest=True,
            )
        )
    else:
        raise NotImplementedError(f'No NLU Model {name}')
