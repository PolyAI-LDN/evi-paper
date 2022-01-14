""" Policy for verification

Copyright PolyAI Limited
"""

from typing import List, Optional, Tuple

import glog

from nlu import Nlu
from readers import Profile, Turn, filter_turns
from verification.models.base import BaseVerificationModel


class VerificationPolicy(object):
    """ Policy for verification """

    def __init__(
        self,
        nlu: Nlu,
        verificator: BaseVerificationModel,
        slot_order: List[str],
        max_attempts: int = 3,
        hard_threshold: float = 0.0
    ):
        """ Initialise

        Args:
            nlu: Nlu
            verificator: the verification model
            slot_order: the slots to use for identification (in order)
            max_attempts: max number of times to ask user about each slot
            hard_threshold: below this threshold, score is set to hard 0.0
        """
        super().__init__()
        assert 0 <= max_attempts <= 3
        self._max_attempts = max_attempts
        #
        self._nlu = nlu
        self._verificator = verificator
        self._slots = slot_order
        self._hard_threshold = hard_threshold
        #
        glog.info('Verification policy initialised!')

    def verify_profile(
        self,
        turns: List[Turn],
        profile: Profile,
    ) -> Optional[Tuple[float, int]]:
        """ Get a verification score for a dialogue and a profile

        Args:
            turns: a list of Turns
            profile: the profile to score for verification

        Returns:
            The verification score as a [0-1] float that indicates
            how well the information in the dialogue matches the
            given profile. The score may be None, if there was not
            conclusive information (e.g. dialogue inputs where empty)
        """
        n_turns = 0
        glog.info(
            f'Verification task: '
            f'dialogue {turns[0].dialogue_id} '
            f'vs '
            f'profile {profile.scenario_id}'
        )
        state = self._verificator.init_state(profile=profile)
        #
        for slot in self._slots:
            for a, t in enumerate(
                filter_turns(
                    turns=turns,
                    item=slot,
                    max_attempts=self._max_attempts
                )
            ):
                if a == 0:
                    state = self._verificator.prepare_item(
                        state=state,
                        item=slot
                    )
                glog.info(f'{slot.upper()} attempt {a + 1}')
                if t.item != slot:
                    glog.error(
                        f'Looked for item {slot.upper()} in wrong turn!'
                    )
                n_turns += 1
                #
                # run NLU
                parsed = self._nlu.run_nlu_for_item(
                    turn=t, item=slot
                )
                # integrate with IDNV
                state = self._verificator.track_item(
                    state=state,
                    parsed=parsed,
                    item=slot
                )
                # # stop after first turn with meaningful hypothesis
                if self._verificator.item_is_collected(
                    state=state,
                    item=slot
                ):
                    break

            if self._hard_threshold > 0.0:
                upper_bound = self._verificator.run_verification(
                    state=state, default=1.0
                )
                if upper_bound <= self._hard_threshold:
                    # stop early!
                    # won't be able to verify even with more inputs
                    return 0.0, n_turns
        #
        score = self._verificator.run_verification(state=state, default=0.0)
        assert 0.0 <= score <= 1.0
        glog.info(f'Verification score={score}')
        return score, n_turns
