""" Models for enrolment

Copyright PolyAI Limited
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from enrolment.models.base import BaseEnrolmentModel, BaseEnrolmentState
from nlu import NluOutput
from nlu.dates import date_to_str
from data_types import Profile, Slot

_NEW_ENROLMENT = 'new_enrolment'


@dataclass()
class SimpleEnrolmentState(BaseEnrolmentState):
    """The state of the enroller"""
    extracted_postcode: str = ''
    extracted_name_first: str = ''
    extracted_name_last: str = ''
    extracted_dob: str = ''


class SimpleEnrolmentModel(BaseEnrolmentModel):
    """ Dummy model for enrolment """

    def __init__(
        self,
        focus_attempt: Optional[int] = None
    ):
        """ Initialise

        Args:
            focus_attempt: set to [1|2|3] to deterministically accept the
                           1st/2nd/3rd collection attempt;
                           use for error analysis/ablation.
        """
        focus_attempt = focus_attempt or 0
        assert focus_attempt == 0 or (1 <= focus_attempt <= 3)
        self.focus_attempt = focus_attempt

    def init_state(self) -> SimpleEnrolmentState:  # noqa: D003
        return SimpleEnrolmentState()

    def track_item(  # noqa: D003
        self,
        state: SimpleEnrolmentState,
        item: str,
        parsed: NluOutput,
    ) -> SimpleEnrolmentState:
        state = deepcopy(state)
        if self.focus_attempt:
            # use this for ablation analysis,
            # where we deterministically accept only the 1st/2nd/3rd attempt
            if state.item2n_attempts[item] - 1 != self.focus_attempt - 1:
                # not the focus attempt / do not track
                return state
        #
        if item == Slot.POSTCODE:
            for p in parsed.postcodes:
                if p:
                    state.extracted_postcode = p
                    break
        elif item == Slot.NAME:
            names = []
            for (first, last) in parsed.names:
                if first and last:
                    names.append((first, last))
            if names:
                state.extracted_name_first = names[0][0]
                state.extracted_name_last = names[0][1]
        elif item == Slot.DOB:
            dates = []
            for d in parsed.dobs:
                d = date_to_str(d)
                dates.append(d)
            if dates:
                state.extracted_dob = dates[0]
        else:
            raise ValueError(f"Unknown item {item}")
        return state

    def item_is_sufficient(  # noqa: D003
        self,
        state: SimpleEnrolmentState,
        item: str,
    ) -> bool:
        if self.focus_attempt:
            # use this for ablation analysis,
            # where we deterministically accept only the 1st/2nd/3rd attempt
            return state.item2n_attempts[item] - 1 == self.focus_attempt - 1
        #
        state = deepcopy(state)
        if item == Slot.POSTCODE:
            return bool(state.extracted_postcode)
        elif item == Slot.NAME:
            return (
                bool(state.extracted_name_first)
                and bool(state.extracted_name_last)
            )
        elif item == Slot.DOB:
            return bool(state.extracted_dob)
        else:
            raise ValueError(f"Unknown item {item}")

    def enroll_profile(  # noqa: D003
        self,
        state: SimpleEnrolmentState,
    ) -> Profile:
        return Profile(
            scenario_id=_NEW_ENROLMENT,
            postcode=state.extracted_postcode,
            name_first=state.extracted_name_first,
            name_last=state.extracted_name_last,
            dob_str=state.extracted_dob,
        )
