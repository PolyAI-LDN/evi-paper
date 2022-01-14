""" Models for NLU

Copyright PolyAI Limited
"""
from __future__ import annotations

from nlu import Nlu
from nlu.dates import EviDateParser
from nlu.names import EviNameParser
from nlu.postcodes import EviPostcodeParser


def build_nlu(name: str, locale: str) -> Nlu:
    """ Build NLU model for the verification task

    Args:
        name: the name of the model
        locale: the locale of the model, e.g. en-US, en-GB

    Returns:
        an NLU model
    """
    locale = locale.replace("_", "-")
    if name == 'risk_averse':
        return Nlu(
            postcode_parser=EviPostcodeParser(
                locale=locale,
                strict=True,
                use_nbest=True,
            ),
            name_parser=EviNameParser(
                locale=locale,
                strict=True,
                use_nbest=True,
            ),
            date_parser=EviDateParser(
                locale=locale,
                strict=True,
                use_nbest=True,
            )
        )
    elif name == 'risk_seeking':
        return Nlu(
            postcode_parser=EviPostcodeParser(
                locale=locale,
                strict=False,
                use_nbest=True,
            ),
            name_parser=EviNameParser(
                locale=locale,
                strict=False,
                use_nbest=True,
            ),
            date_parser=EviDateParser(
                locale=locale,
                strict=False,
                use_nbest=True,
            )
        )
    else:
        raise NotImplementedError(f'No NLU Model {name}')
