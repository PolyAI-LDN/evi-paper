""" Models for identification

Copyright PolyAI Limited
"""
from __future__ import annotations

from identification.models.base import BaseIdentificationModel
from identification.models.fallback import FallbackIdentificationModel
from utils.logic import Logic
from utils.similarity import ConstantScorer, ExactScorer, TextEditScorer


def build_model_i(name: str, locale: str) -> BaseIdentificationModel:
    """ Create a new identification model

    Args:
        name: the name of the model
        locale: the locale of the model (language_Region)

    Returns:
        an identification model
    """
    locale = locale.replace("_", "-")
    if name in {"oracle"}:
        # load a dummy model - will not be used
        name = "exact-1"
    elif name in {"none"}:
        name = "none-1"
    name, alpha = name.split("-")
    alpha = float(alpha)
    #
    if name == 'none':
        return FallbackIdentificationModel(
            postcode_scorer=ConstantScorer(),
            name_scorer=ConstantScorer(),
            date_scorer=ConstantScorer(),
            logic=Logic(alpha=alpha),
        )
    elif name == 'exact':
        return FallbackIdentificationModel(
            postcode_scorer=ExactScorer(),
            name_scorer=ExactScorer(),
            date_scorer=ExactScorer(),
            logic=Logic(alpha=alpha),
        )
    elif name == 'fuzzy':
        return FallbackIdentificationModel(
            postcode_scorer=TextEditScorer(),
            name_scorer=TextEditScorer(),
            date_scorer=TextEditScorer(),
            logic=Logic(alpha=alpha),
        )
    raise NotImplementedError(f'No Identification Model {name}')
