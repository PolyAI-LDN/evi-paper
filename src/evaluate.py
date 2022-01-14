""" Run models and evaluations for tasks

python projects/evi/evaluate.py \
    -eiv \
    --locale en_GB \
    --model naive

Copyright PolyAI Limited
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List

import glog
from numpy.random import RandomState

from enrolment import EnrolmentPolicy, build_model_e
from evaluators import (
    EnrolmentEvaluator, IdentificationEvaluator, VerificationEvaluator
)
from identification import IdentificationPolicy, build_model_i
from identification.api import ProfileDatastore
from nlu import build_nlu
from readers import Profile, Slot, Turn, read_dialogues, read_profiles
from verification import VerificationPolicy, build_model_v

_DATA_DIR = "../data"


def _randomized_verification_attempts(
    dialogue_id2turns: Dict[str, List[Turn]],
    scenario_id2profile: Dict[str, Profile],
    rate_genuine_to_impostor: float,
    seed=123
) -> Dict[str, List[str]]:
    """ Return dict from dialogue id to list of scenario ids of attempts """
    # TODO store in file?
    # TODO random vs adversarial choice of attempts
    rng = RandomState(seed=seed)
    n_attempts_genuine = 0
    n_attempts_impostor = 0
    dialogue_id2scenario_ids = defaultdict(list)
    all_profiles = [
        p
        for scenario_id, p in sorted(
            list(scenario_id2profile.items()),
            key=lambda x: x[0]
        )
    ]
    #
    for dialogue_id, turns in sorted(
        list(dialogue_id2turns.items()),
        key=lambda x: x[0]
    ):
        target_scenario_id = turns[0].scenario_id
        # genuine attempt
        try:
            profile = scenario_id2profile[target_scenario_id]
            attempt_scenario_id = profile.scenario_id
            dialogue_id2scenario_ids[dialogue_id].append(attempt_scenario_id)
            n_attempts_genuine += 1
        except KeyError:
            pass
        # impostor attempts
        n_impostors = int(round(
            n_attempts_genuine / rate_genuine_to_impostor - n_attempts_impostor
        ))
        for i in rng.permutation(len(all_profiles))[:n_impostors]:
            attempt_scenario_id = all_profiles[i].scenario_id
            if attempt_scenario_id == target_scenario_id:
                # skip genuine attempts
                continue
            dialogue_id2scenario_ids[dialogue_id].append(attempt_scenario_id)
            n_attempts_impostor += 1
    n_attempts_total = n_attempts_genuine + n_attempts_impostor
    glog.info(f'Built randomised verification attempts '
              f'(Genuine: {n_attempts_genuine} '
              f'Impostor: {n_attempts_impostor} '
              f'Total: {n_attempts_total})')
    return dict(dialogue_id2scenario_ids)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--locale', type=str, default='en_GB',
        choices={"en_GB", "pl_PL", "fr_FR", "el_GR"},
        help='Locale to evaluate'
    )
    parser.add_argument(
        '-e', '--enrolment',
        action='store_true',
        help='Whether to evaluate enrolment'
    )
    parser.add_argument(
        '-v', '--verification',
        action='store_true',
        help='Whether to evaluate verification'
    )
    parser.add_argument(
        '-i', '--identification',
        action='store_true',
        help='Whether to evaluate identification'
    )
    parser.add_argument(
        '--nlu',
        default='risk_averse',
        choices={
            "risk_averse", "risk_seeking",
        },
        help='Which NLU model to evaluate'
    )
    parser.add_argument(
        '--model',
        type=str, default='exact',
        choices={
            # for Enrolment
            "0",
            "1", "2", "3",
            # for Verification
            "random",
            "exact", "fuzzy",
            # for Identification
            "oracle", "none",
            "exact-1", "exact-0.5",
            "fuzzy-1", "fuzzy-0.5",
        },
        help='Which model to evaluate'
    )
    parser.add_argument(
        '-t', '--thresh',
        type=float,
        default=0.0,
        help='Hard threshold (below this, set verification score to zero)'
    )
    parser.add_argument(
        '-p', '--plots',
        action='store_true',
        help='Whether to show plots'
    )
    parser.add_argument(
        '--eager',
        action='store_true',
        help='Whether to terminate identification asap'
    )
    parser.add_argument(
        '--kbo',
        action='store_true',
        help='Whether to use the KB oracle for identification'
    )
    args = parser.parse_args()
    #
    if not (args.enrolment or args.identification or args.verification):
        parser.error('Not specified task to evaluate; add: [-e] [-i] [-v]')

    #
    locale = args.locale.replace("_", "-")
    lang_code = locale.split('-')[0]
    #
    profiles_file = os.path.join(
        _DATA_DIR, f"records.{lang_code}.csv"
    )
    dialogues_file = os.path.join(
        _DATA_DIR, f"dialogues.{lang_code}.tsv"
    )
    slot_order = [Slot.POSTCODE, Slot.NAME, Slot.DOB]
    #
    scenario_id2profile = read_profiles(profiles_file)
    dialogue_id2turns = read_dialogues(dialogues_file)
    print(f'Read {len(scenario_id2profile)} profiles')
    print(f'Read {len(dialogue_id2turns)} dialogues')
    if args.enrolment:
        policy_e = EnrolmentPolicy(
            nlu=build_nlu(name=args.nlu, locale=locale),
            enroller=build_model_e(name=args.model),
            slot_order=slot_order,
        )
        evaluator_e = EnrolmentEvaluator(
            slots=slot_order
        )
        for dialogue_id, turns in dialogue_id2turns.items():
            gold_scenario_id = turns[0].scenario_id
            gold_profile = scenario_id2profile.get(gold_scenario_id, None)
            if not gold_profile:
                glog.warn(
                    f'Skipping scenario not in DB: {gold_scenario_id}'
                )
                continue
            pred_profile, slot2n_attempts = policy_e.enrol_profile(turns=turns)
            evaluator_e.increment(
                pred_profile=pred_profile,
                gold_profile=gold_profile,
                slot2n_attempts=slot2n_attempts
            )
        evaluator_e.print_report()
    #
    if args.identification:
        policy_i = IdentificationPolicy(
            db=ProfileDatastore(
                scenario_id2profile,
                searchable_by_postcode=True,
                searchable_by_name=False,
                searchable_by_dob=False,
                searchable_by_floa=True,
                return_all_profiles=False,
            ),
            nlu=build_nlu(name=args.nlu, locale=locale),
            identificator=build_model_i(
                name=args.model,
                locale=locale
            ),
            slot_order=slot_order,
            eager_identification=args.eager,
            use_db_oracle=args.kbo,
            use_i_oracle=args.model == "oracle",
        )
        evaluator_i = IdentificationEvaluator(
            scenario_id2profile=scenario_id2profile,
            dialogue_id2turns=dialogue_id2turns
        )
        n_unk_scenario = 0
        for i, (dialogue_id, turns) in enumerate(dialogue_id2turns.items()):
            glog.info(f'Seen {i} dialogues')
            gold_scenario_id = turns[0].scenario_id
            if gold_scenario_id is None:
                glog.warn(
                    f"Skipping with unknown scenario: {dialogue_id}"
                )
                n_unk_scenario += 1
                continue
            gold_profile = scenario_id2profile.get(gold_scenario_id, None)
            if not gold_profile:
                glog.warn(
                    f'Skipping scenario not in DB: {gold_scenario_id}'
                )
                continue
            pred_profile_id, n_turns = policy_i.identify_profile(turns=turns)
            evaluator_i.increment(
                pred_id=pred_profile_id,
                gold_id=gold_scenario_id,
                dialogue_id=dialogue_id,
                n_turns=n_turns
            )
        print(f'Skipped {n_unk_scenario} dialogues with unknown scenario')
        evaluator_i.print_report()
    #
    if args.verification:
        policy_v = VerificationPolicy(
            nlu=build_nlu(name=args.nlu, locale=locale),
            verificator=build_model_v(args.model, locale=locale),
            slot_order=slot_order,
            hard_threshold=args.thresh,
        )
        evaluator_v = VerificationEvaluator()
        #
        dialogue_id2scenario_ids = _randomized_verification_attempts(
            dialogue_id2turns=dialogue_id2turns,
            scenario_id2profile=scenario_id2profile,
            rate_genuine_to_impostor=1 / 1
        )
        #
        for dialogue_id, turns in dialogue_id2turns.items():
            for scenario_id in dialogue_id2scenario_ids.get(dialogue_id, []):
                profile = scenario_id2profile[scenario_id]
                score, n_turns = policy_v.verify_profile(
                    turns=turns,
                    profile=profile
                )
                evaluator_v.increment(
                    pred_score=score,
                    is_genuine=(turns[0].scenario_id == profile.scenario_id),
                    n_turns=n_turns
                )
        evaluator_v.print_report(args.plots)
    #
    print(args)


if __name__ == '__main__':
    _main()
    glog.info("Done!")
