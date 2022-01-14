""" Abstract models for enrolment

Copyright PolyAI Limited
"""
from __future__ import annotations

import abc
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict

from data_types import Profile
from nlu import NluOutput


class BaseEnrolmentModel(abc.ABC):
    """ Abstract model for enrolment """

    @abc.abstractmethod
    def init_state(self) -> BaseEnrolmentState:
        """ Initialise the enrolment state """
        raise NotImplementedError()

    def prepare_item(
        self,
        state: BaseEnrolmentState,
        item: str,
    ) -> BaseEnrolmentState:
        """ Prepare to track item

        Args:
            state: the state of the enroller
            item: an item to be enrolled

        Returns:
            the updated state
        """
        state = deepcopy(state)
        try:
            state.item2n_attempts[item] += 1
        except KeyError:
            state.item2n_attempts[item] = 1
        return state

    @abc.abstractmethod
    def track_item(
        self,
        state: BaseEnrolmentState,
        item: str,
        parsed: NluOutput,
    ) -> BaseEnrolmentState:
        """ The actions to keep track of an item

        Args:
            state: the state of the enroller
            item: an item to be enrolled
            parsed: the NLU output for the current turn turn
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def item_is_sufficient(
        self,
        state: BaseEnrolmentState,
        item: str,
    ) -> bool:
        """ Whether the enroller has confidently collected an item

        The definition of when the model is satisfied with the collection
        of an item's values is a design decision

        Args:
            state: the state of the enroller
            item: the item to be enrolled
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def enroll_profile(
        self,
        state: BaseEnrolmentState,
    ) -> Profile:
        """ Return the enrolled profile

        Args:
            state: the state of the enroller
        """
        raise NotImplementedError()


@dataclass()
class BaseEnrolmentState(abc.ABC):
    """ Basis for the state of the enroller """
    item2n_attempts: Dict[str, int] = field(default_factory=lambda: {})
