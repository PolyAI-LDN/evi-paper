""" Policy for enrolment

Copyright PolyAI Limited
"""
from typing import Dict, List, Tuple

import glog

from enrolment.models.base import BaseEnrolmentModel
from nlu import Nlu
from data_types import Profile, Turn, filter_turns


class EnrolmentPolicy(object):
    """ Enrolment policy """

    def __init__(
        self,
        nlu: Nlu,
        enroller: BaseEnrolmentModel,
        slot_order: List[str],
        max_attempts: int = 3,
    ):
        """ Initialise

        Args:
            nlu: Nlu model
            enroller: enrollment model
            slot_order: the slots to use for identification (in order)
            max_attempts: max number of times to ask user about each slot
        """
        assert 0 <= max_attempts <= 3
        self._max_attempts = max_attempts
        self._nlu = nlu
        self._enroller = enroller
        self._slots = slot_order
        #
        glog.info('Enrolment policy initialised!')

    def enrol_profile(
        self,
        turns: List[Turn]
    ) -> Tuple[Profile, Dict[str, int]]:
        """ Get profile extracted from dialogue and how many turns it took

        Args:
            turns: a list of turns of the dialogue

        Returns:
            a tuple Profile, item2n_attempts (dict with how many turns
            were required to extract each slot)
        """
        state = self._enroller.init_state()
        for slot in self._slots:
            for i, t in enumerate(filter_turns(
                turns=turns,
                item=slot,
                max_attempts=self._max_attempts
            )):
                if t.item != slot:
                    glog.error(
                        f'Looked for item {slot.upper()} in wrong turn!'
                    )
                # run nlu
                parsed = self._nlu.run_nlu_for_item(turn=t, item=slot)
                state = self._enroller.prepare_item(state=state, item=slot)
                state = self._enroller.track_item(
                    state=state,
                    item=slot,
                    parsed=parsed
                )
                #
                if self._enroller.item_is_sufficient(
                    state=state,
                    item=slot,
                ):
                    break
        #
        return (
            self._enroller.enroll_profile(state=state),
            state.item2n_attempts
        )
