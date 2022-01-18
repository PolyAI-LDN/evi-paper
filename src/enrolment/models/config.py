""" Models for enrolment

Copyright PolyAI Limited
"""
from __future__ import annotations

from enrolment.models.base import BaseEnrolmentModel
from enrolment.models.simple import SimpleEnrolmentModel


def build_model_e(name: str) -> BaseEnrolmentModel:
    """ Create a new enrolment model """
    name = name.strip()
    if name == '0':
        return SimpleEnrolmentModel(
            focus_attempt=None
        )
    elif name in {'1', '2', '3'}:
        return SimpleEnrolmentModel(
            focus_attempt=int(name)
        )
    else:
        raise NotImplementedError(f'No enrolment model {name}')
