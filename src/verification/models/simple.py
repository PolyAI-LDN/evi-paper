""" Simple model for verification

Copyright PolyAI Limited
"""
from __future__ import annotations

import datetime as dt
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from data_types import Profile, Slot
from nlu import NluOutput
from utils.logic import Logic
from utils.scoring import BaseScorer
from verification.models.base import (
    BaseVerificationModel, BaseVerificationState
)


@dataclass()
class SimpleVerificationState(BaseVerificationState):
    """ Basis for the state of the verificator """
    postcode_guesses: List[str] = field(default_factory=lambda: [])
    name_full_guesses: List[str] = field(default_factory=lambda: [])
    name_parts_guesses: List[Tuple[str, str]] = field(
        default_factory=lambda: []
    )
    dob_guesses: List[dt.date] = field(default_factory=lambda: [])
    #
    postcode_score: Optional[float] = None
    name_full_score: Optional[float] = None
    name_parts_scores: List[Tuple[float, float]] = field(
        default_factory=lambda: []
    )
    dob_score: Optional[float] = None


class SimpleVerificationModel(BaseVerificationModel):
    """ Simple model for verification """

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

    def init_state(  # noqa: D003
        self,
        profile: Profile
    ) -> SimpleVerificationState:
        return SimpleVerificationState(profile=profile)

    def prepare_item(  # noqa: D003
        self,
        state: SimpleVerificationState,
        item: str,
    ) -> SimpleVerificationState:
        state = deepcopy(state)
        if item in {
            Slot.POSTCODE,
            Slot.NAME,
            Slot.DOB,
        }:
            return state
        else:
            raise ValueError(f"Unknown item {item}")

    def track_item(  # noqa: D003
        self,
        state: SimpleVerificationState,
        parsed: NluOutput,
        item: str,
    ) -> SimpleVerificationState:
        state = deepcopy(state)
        assert state.profile is not None
        if item == Slot.POSTCODE:
            if parsed.postcodes:
                state.postcode_guesses.extend(parsed.postcodes)
                state.postcode_score = self._postcode_scorer.calc_score(
                    hyps=state.postcode_guesses,
                    ref=state.profile.postcode
                )
            return state
        elif item == Slot.NAME:
            names = parsed.get_full_names()
            if names:
                state.name_full_guesses.extend(names)
                state.name_full_score = self._name_scorer.calc_score(
                    hyps=state.name_full_guesses,
                    ref=state.profile.name_full
                )
            if parsed.names:
                state.name_parts_guesses.extend(parsed.names)
                for first, last in parsed.names:
                    if not first or not last:
                        continue
                    score_first = self._name_scorer.calc_score(
                        hyps=[first],
                        ref=state.profile.name_first
                    )
                    score_last = self._name_scorer.calc_score(
                        hyps=[last],
                        ref=state.profile.name_last
                    )
                    state.name_parts_scores.append((score_first, score_last))
            return state
        elif item == Slot.DOB:
            if parsed.dobs:
                state.dob_guesses.extend(parsed.dobs)
                state.dob_score = self._date_scorer.calc_score(
                    hyps=state.dob_guesses,
                    ref=state.profile.dob
                )
            return state
        else:
            raise ValueError(f"Unknown item {item}")

    def item_is_collected(  # noqa: D003
        self,
        state: SimpleVerificationState,
        item: str,
    ) -> bool:
        if item == Slot.POSTCODE:
            return state.postcode_score is not None
        elif item == Slot.NAME:
            return state.name_full_score is not None
        elif item == Slot.DOB:
            return state.dob_score is not None
        else:
            raise NotImplementedError(f"Unknown item {item}")

    def run_verification(  # noqa: D003
        self,
        state: SimpleVerificationState,
        default: float = 1.0
    ) -> float:
        score = self._logic.combine_scores(
            score_postcode=_resolve_none(state.postcode_score, default),
            score_name_full=_resolve_none(state.name_full_score, default),
            scores_name_parts=state.name_parts_scores,
            score_dob=_resolve_none(state.dob_score, default),
        )
        return score


def _resolve_none(x: float, default: float) -> float:
    if x is None:
        return default
    return x
