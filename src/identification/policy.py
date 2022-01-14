""" Policy for identification

Copyright PolyAI Limited
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import md5
from typing import List, Tuple

import glog

from identification.api import ProfileDatastore
from identification.models import BaseIdentificationModel
from nlu import Nlu, NluOutput
from data_types import Profile, Slot, Turn, filter_turns

_NO_PROFILE = 'Identified.None'


class IdentificationPolicy(object):
    """ The policy for performing profile identification """

    def __init__(
        self,
        db: ProfileDatastore,
        nlu: Nlu,
        identificator: BaseIdentificationModel,
        slot_order: List[str],
        max_attempts: int = 3,
        eager_identification: bool = True,
        use_db_oracle: bool = False,
        use_i_oracle: bool = False
    ):
        """ Initialise

        Args:
            db: the profile datastore
            nlu: Nlu model
            identificator: the identification model
            slot_order: the slots to use for identification (in order)
            max_attempts: max number of times to ask user about each slot
            eager_identification: whether to stop asap a profile is identified
            use_db_oracle: whether to force API results to ALWAYS contain the
                           target users (regardless of user input)
            use_i_oracle: whether to force identificator results to ALWAYS be
                          correct (regardless of user input) when the correct
                          target profile is being tracked
        """
        assert 0 <= max_attempts <= 3
        self._max_attempts = max_attempts
        self._eager_identification = eager_identification
        self._use_db_oracle = use_db_oracle
        self._use_i_oracle = use_i_oracle
        #
        self._db = db
        self._nlu = nlu
        self._identificator = identificator
        self._slots = slot_order
        glog.info('Identification policy initialised!')

    @lru_cache()
    def _can_search_profiles_after_item(self, item: str):
        """ Whether users can be collected after (exclusive) item turn"""
        answer = False
        after_item = False
        for slot in self._slots:
            if item == slot:
                after_item = True
                continue
            if after_item:
                answer = answer or self._db.is_searchable_by(item=slot)
        return answer

    def _query_for_profiles(
        self,
        parsed: NluOutput,
        item: str
    ) -> List[Profile]:
        profiles = []
        if item == Slot.POSTCODE:
            for hyp in parsed.postcodes:
                profiles.extend(self._db.find_by_postcode(hyp))
        elif item == Slot.NAME:
            for first, last in parsed.names:
                profiles.extend(self._db.find_by_name(f"{first} {last}"))
        elif item == Slot.DOB:
            for hyp in parsed.dobs:
                profiles.extend(self._db.find_by_dob(hyp))
        else:
            raise ValueError(f"Unknown item {item}")
        return profiles

    def identify_profile(
        self,
        turns: List[Turn],
    ) -> Tuple[str, int]:
        """ Execute the identification policy

        Args:
            turns: list of turns

        Returns:
            a tuple of the identified profile ID and the number of input turns
            needed for performing identification
        """
        glog.info(
            f'Identification task: '
            f'dialogue {turns[0].dialogue_id}'
        )
        n_turns = 0
        state = self._identificator.init_state()
        for _i_slot, slot in enumerate(self._slots):
            if (
                _i_slot != 0
                and not self._identificator.has_profiles(state)
                and not self._db.is_searchable_by(item=slot)
                and not self._can_search_profiles_after_item(item=slot)
            ):
                # has not tracked any profiles yet
                # and cannot track any profiles in the future
                # -> stop early
                return _NO_PROFILE, n_turns
            for a, t in enumerate(
                filter_turns(
                    turns=turns,
                    item=slot,
                    max_attempts=self._max_attempts
                )
            ):
                glog.info(f'{slot.upper()} attempt {a + 1}')
                if a == 0:
                    state = self._identificator.prepare_item(
                        state=state,
                        item=slot
                    )
                n_turns += 1
                if t.item != slot:
                    glog.error(
                        f'Looked for item {slot.upper()} in wrong turn!'
                    )
                # run NLU
                parsed = self._nlu.run_nlu_for_item(
                    turn=t,
                    item=slot
                )
                # query API
                profiles = self._query_for_profiles(
                    parsed=parsed,
                    item=slot
                )
                if self._use_db_oracle:
                    # oracle to simulate a "perfect" API
                    # that always returns the target profile too
                    profiles.extend(
                        self._db.find_by_oracle(
                            scenario_id=t.scenario_id
                        )
                    )
                profiles = _reorder_profiles(_filter_profiles(profiles))

                # integrate with IDNV
                state = self._identificator.track_profiles(
                    state=state,
                    profiles=profiles
                )
                state = self._identificator.track_item(
                    state=state,
                    item=slot,
                    parsed=parsed,
                )
                #
                if self._use_i_oracle:
                    identified = self._identificator.run_oracle(
                        scenario_id=t.scenario_id,
                        state=state
                    )
                    if identified:
                        return identified[0], n_turns
                else:
                    if (
                        self._eager_identification
                        and not self._can_search_profiles_after_item(item=slot)
                    ):
                        # try to identify user asap
                        identified = self._identificator.eager_identification(
                            state=state
                        )
                        if identified:
                            return identified[0], n_turns
                #
                if (
                    # item is tracked
                    self._identificator.item_is_sufficient(state, slot)
                    # profiles are tracked (or can't be retrieved) for item
                    and (
                        self._identificator.profiles_are_sufficient(state)
                        or not self._db.is_searchable_by(item=slot)
                    )
                ):
                    break
        #
        if not self._identificator.has_profiles(state):
            return _NO_PROFILE, n_turns
        # final run of id algo
        if self._use_i_oracle:
            identified = self._identificator.run_oracle(
                scenario_id=turns[-1].scenario_id,
                state=state
            )
        else:
            identified = self._identificator.run_identification(
                state=state
            )
        if identified:
            return identified[0], n_turns
        return _NO_PROFILE, n_turns


def _reorder_profiles(profiles: List[Profile]) -> List[Profile]:
    # shuffle retrieved profiles,
    # so that order is not indicative of ground truth
    return sorted(
        profiles, key=lambda p: int(
            md5(p.scenario_id.encode("utf-8")).hexdigest()[:6],
            16
        )
    )


def _filter_profiles(profiles: List[Profile]) -> List[Profile]:
    unique_ids = set()
    unique_profiles = []
    for p in profiles:
        if p.scenario_id in unique_ids:
            continue
        unique_ids.add(p.scenario_id)
        unique_profiles.append(p)
    return unique_profiles
