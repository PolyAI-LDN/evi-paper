""" Abstract models for identification

Copyright PolyAI Limited
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List

from nlu import NluOutput
from data_types import Profile


class BaseIdentificationModel(abc.ABC):
    """ Abstract model for identification """

    @abc.abstractmethod
    def init_state(self) -> BaseIdentificationState:
        """ Initialise the identificator's state """
        raise NotImplementedError()

    @abc.abstractmethod
    def prepare_item(
        self,
        state: BaseIdentificationState,
        item: str,
    ) -> BaseIdentificationState:
        """ Any actions BEFORE collecting an item

        Args:
            state: the state of the identificator
            item: an identification item
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def track_item(
        self,
        state: BaseIdentificationState,
        item: str,
        parsed: NluOutput,
    ) -> BaseIdentificationState:
        """ The actions to keep track of an item

        Args:
            state: the state of the identificator
            item: an identification item
            parsed: the NLU output for the current turn turn
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def track_profiles(
        self,
        state: BaseIdentificationState,
        profiles: List[Profile],
    ) -> BaseIdentificationState:
        """ The actions to keep track of a list of candidate profiles

        Args:
            state: the state of the identificator
            profiles: a list of candidate profiles
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def item_is_sufficient(
        self,
        state: BaseIdentificationState,
        item: str,
    ) -> bool:
        """Whether the identificator is confidently tracking an item

        The definition of when the model is satisfied with the collection
        of an item's values is a design decision

        Args:
            state: the state of the identificator
            item: an identification item
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def profiles_are_sufficient(
        self,
        state: BaseIdentificationState,
    ) -> bool:
        """Whether the identificator is confidently tracking candidate profiles

        The definition of when the model is satisfied with the collection
        of candidate profiles is a design decision

        Args:
            state: the state of the identificator
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_identification(
        self,
        state: BaseIdentificationState
    ) -> List[str]:
        """ Return a list of most probable profile ids

        Args:
            state: the state of the identificator
        """
        raise NotImplementedError()

    def eager_identification(
        self,
        state: BaseIdentificationState,
    ) -> List[str]:
        """ Return a list of most probable profile ids for early stopping

        This is to run eager identification and return anytime results.

        Args:
            state: the state of the identificator

        Returns:
             a list of identified profiles
        """
        identified = self.run_identification(state)
        return identified

    def run_oracle(
        self,
        scenario_id: str,
        state: BaseIdentificationState
    ) -> List[str]:
        """ Oracle to simulate a "perfect" identification

        It always returns the target profile,
        IF the target profile is being tracked

        Args:
            scenario_id: ground truth scenario id
            state: the state of the identificator

        Returns:
            a list with the correct target profile, IF it was tracked
        """
        # oracle to simulate a "perfect" identification
        # that always returns the target profile,
        # IF it is being tracked
        if scenario_id in self.list_profile_ids(
            state=state
        ):
            return [scenario_id]
        return []

    @abc.abstractmethod
    def list_profile_ids(
        self,
        state: BaseIdentificationState,
    ) -> List[str]:
        """ Return a list of tracked profile ids

        Args:
            state: the state of the identificator
        """
        raise NotImplementedError()

    def has_profiles(
        self,
        state: BaseIdentificationState,
    ) -> bool:
        """ Whether the identificator is tracking any candidate profiles """
        return bool(self.list_profile_ids(state))


@dataclass()
class BaseIdentificationState(abc.ABC):
    """ Basis for the identificator state """
    pass
