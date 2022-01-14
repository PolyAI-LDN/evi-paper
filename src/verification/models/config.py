""" Models for verification

Copyright PolyAI Limited
"""
from __future__ import annotations

from utils.logic import Logic
from utils.similarity import ExactScorer, RandomScorer, TextEditScorer
from verification.models.base import BaseVerificationModel
from verification.models.simple import SimpleVerificationModel


def build_model_v(name: str, locale: str) -> BaseVerificationModel:
    """ Create a new verification model

    Args:
        name: the name of the model
        locale: the locale of the model (language_Region)

    Returns:
        a verification model
    """
    locale = locale.replace("_", "-")
    if name == 'random':
        return SimpleVerificationModel(
            postcode_scorer=RandomScorer(),
            name_scorer=RandomScorer(),
            date_scorer=RandomScorer(),
            logic=Logic()
        )
    elif name == 'exact':
        return SimpleVerificationModel(
            postcode_scorer=ExactScorer(),
            name_scorer=ExactScorer(),
            date_scorer=ExactScorer(),
            logic=Logic()
        )
    elif name == 'fuzzy':
        return SimpleVerificationModel(
            postcode_scorer=TextEditScorer(),
            name_scorer=TextEditScorer(),
            date_scorer=TextEditScorer(),
            logic=Logic()
        )
    else:
        raise NotImplementedError(f'No verification Model {name}')
