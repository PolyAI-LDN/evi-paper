""" Simple model for identification

Copyright PolyAI Limited
"""
from __future__ import annotations

import datetime as dt
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple

from data_types import Profile, Slot
from identification.models.base import (
    BaseIdentificationModel, BaseIdentificationState
)
from nlu import NluOutput
from utils.logic import Logic
from utils.similarity import BaseScorer


@dataclass()
class FallbackIdentificationState(BaseIdentificationState):
    """ The state of the identificator """
    profiles: List[Profile] = field(default_factory=lambda: [])
    postcode_guesses: List[str] = field(default_factory=lambda: [])
    name_full_guesses: List[str] = field(default_factory=lambda: [])
    name_parts_guesses: List[Tuple[str, str]] = field(
        default_factory=lambda: []
    )
    dob_guesses: List[dt.date] = field(default_factory=lambda: [])


class FallbackIdentificationModel(BaseIdentificationModel):
    """ Very naive model for identification (for dev purposes) """

    def __init__(
        self,
        postcode_scorer: BaseScorer,
        name_scorer: BaseScorer,
        date_scorer: BaseScorer,
        logic: Logic,
    ):
        """ Initialise

        Args:
            postcode_scorer: a scorer for verification matching of postcodes
            name_scorer: a scorer for verification matching of names
            date_scorer: a scorer for verification matching of dates
            logic: the logic operators to combine verification scores
        """
        self._postcode_scorer = postcode_scorer
        self._name_scorer = name_scorer
        self._date_scorer = date_scorer
        self._logic = logic

    def init_state(self) -> FallbackIdentificationState:  # noqa: D003
        return FallbackIdentificationState()

    def prepare_item(  # noqa: D003
        self,
        state: FallbackIdentificationState,
        item: str,
    ) -> FallbackIdentificationState:
        state = deepcopy(state)
        return state

    def track_item(  # noqa: D003
        self,
        state: FallbackIdentificationState,
        item: str,
        parsed: NluOutput,
    ) -> FallbackIdentificationState:
        state = deepcopy(state)
        if item == Slot.POSTCODE:
            state.postcode_guesses.extend(parsed.postcodes)
        elif item == Slot.NAME:
            state.name_full_guesses.extend(parsed.get_full_names())
            state.name_parts_guesses.extend(parsed.names)
        elif item == Slot.DOB:
            state.dob_guesses.extend(parsed.dobs)
        else:
            raise ValueError(f"Unknown item {item}")
        return state

    def track_profiles(  # noqa: D003
        self,
        state: FallbackIdentificationState,
        profiles: List[Profile],
    ) -> FallbackIdentificationState:
        state = deepcopy(state)
        all_ids = set()
        for p in state.profiles:
            all_ids.add(p.scenario_id)
        for p in profiles:
            if p.scenario_id not in all_ids:
                state.profiles.append(p)
                all_ids.add(p.scenario_id)
        return state

    def _compute_profile_score(
        self,
        profile: Profile,
        state: FallbackIdentificationState
    ) -> float:
        # compute per-item scores
        postcode_score = self._postcode_scorer.calc_score(
            hyps=state.postcode_guesses,
            ref=profile.postcode
        )
        name_full_score = self._name_scorer.calc_score(
            hyps=state.name_full_guesses,
            ref=profile.name_full
        )
        name_parts_scores = []
        for first, last in state.name_parts_guesses:
            if not first or not last:
                continue
            score_first = self._name_scorer.calc_score(
                hyps=[first],
                ref=profile.name_first
            )
            score_last = self._name_scorer.calc_score(
                hyps=[last],
                ref=profile.name_last
            )
            name_parts_scores.append((score_first, score_last))
        dob_score = self._date_scorer.calc_score(
            hyps=state.dob_guesses,
            ref=profile.dob
        )
        # compute whole-profile score
        return self._logic.combine_scores(
            score_postcode=postcode_score,
            score_name_full=name_full_score,
            scores_name_parts=name_parts_scores,
            score_dob=dob_score,
        )

    def run_identification(  # noqa: D003
        self,
        state: FallbackIdentificationState
    ) -> List[str]:
        state = deepcopy(state)
        profiles = state.profiles
        #
        profiles_with_scores = []
        for i, p in enumerate(profiles):
            score = self._compute_profile_score(
                profile=p,
                state=state,
            )
            if score > 0:
                profiles_with_scores.append((p, score, i))
        profiles_with_scores = sorted(
            profiles_with_scores,
            key=lambda x: (-x[1], x[2])
        )
        retrieved = [x[0].scenario_id for x in profiles_with_scores]
        return retrieved

    def item_is_sufficient(  # noqa
        self,
        state: FallbackIdentificationState,
        item: str,
    ) -> bool:
        """ Done collecting values for an item """
        if item == Slot.POSTCODE:
            return bool(state.postcode_guesses)
        elif item == Slot.NAME:
            return bool(state.name_full_guesses)
        elif item == Slot.DOB:
            return bool(state.dob_guesses)
        raise NotImplementedError(f'Unknown slot {item}')

    def profiles_are_sufficient(  # noqa: D003
        self,
        state: FallbackIdentificationState,
    ) -> bool:
        """ Done collecting profiles """
        return bool(state.profiles)

    def list_profile_ids(  # noqa: D003
        self,
        state: FallbackIdentificationState,
    ) -> List[str]:
        return [p.scenario_id for p in state.profiles]


def _normalise_postcode(s: str) -> str:
    return s.replace(' ', '').replace("-", "").upper()


def _normalise_name(s: str) -> str:
    return s.lower()
