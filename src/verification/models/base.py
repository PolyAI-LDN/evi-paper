""" Base model for verification

Copyright PolyAI Limited
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

from nlu import NluOutput
from data_types import Profile


class BaseVerificationModel(abc.ABC):
    """ Abstract model for verification """

    @abc.abstractmethod
    def init_state(self, profile: Profile) -> BaseVerificationState:
        """ Initialise the state of the verificator

        Args:
            profile: the profile to be verified
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def prepare_item(
        self,
        state: BaseVerificationState,
        item: str,
    ) -> BaseVerificationState:
        """Any actions BEFORE collecting an item

        Args:
            state: the state of the verificator
            item: an identification item
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def track_item(
        self,
        state: BaseVerificationState,
        parsed: NluOutput,
        item: str,
    ) -> BaseVerificationState:
        """ The actions to keep track of an item

        Args:
            state: the state of the identificator
            parsed: the NLU output for the current turn turn
            item: an identification item
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def item_is_collected(
        self,
        state: BaseVerificationState,
        item: str,
    ) -> bool:
        """Whether the verificator is confidently tracking an item

        Args:
            state: the state of the verificator
            item: an identification item
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_verification(
        self,
        state: BaseVerificationState,
        default: float
    ) -> float:
        """ Return a verification [0-1] score

        Args:
            state: the state of the verificator
            default: default value if an item's score is None
        """
        raise NotImplementedError()


@dataclass()
class BaseVerificationState(abc.ABC):
    """ Basis for the state of the verificator """
    profile: Profile
