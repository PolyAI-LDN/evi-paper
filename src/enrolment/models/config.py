""" Models for enrolment

Copyright PolyAI Limited
"""
from __future__ import annotations

from enrolment.models.base import BaseEnrolmentModel
from enrolment.models.simple import SimpleEnrolmentModel


def build_model_e(name: str) -> BaseEnrolmentModel:
    """ Create a new enrolment model """
    name = name.strip()
    if name in {'0', 'naive', 'basic'}:
        focus_attempt = None
    else:
        focus_attempt = int(name)
        assert 1 <= focus_attempt <= 3
    return SimpleEnrolmentModel(
        focus_attempt=focus_attempt
    )
