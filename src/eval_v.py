""" Run models and evaluations for tasks

python projects/evi/eval_e.py \
    -eiv \
    --locale en_GB \
    --model naive

Copyright PolyAI Limited
"""

import argparse
from collections import defaultdict
from typing import Dict, List

import glog
from numpy.random import RandomState

from nlu import build_nlu
from data_types import Profile, Slot, Turn
from readers import read_evi_data
from verification import (
    VerificationEvaluator, VerificationPolicy, build_model_v
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--locale', type=str, default='en_GB',
        choices={"en_GB", "pl_PL", "fr_FR"},
        help='Locale to evaluate'
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
        choices={"random","exact", "fuzzy"},
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
    return parser.parse_args()


def _randomized_verification_attempts(
    dialogue_id2turns: Dict[str, List[Turn]],
    scenario_id2profile: Dict[str, Profile],
    rate_genuine_to_impostor: float,
    seed=123
) -> Dict[str, List[str]]:
    """ Return dict from dialogue id to list of scenario ids of attempts """
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
    args = parse_args()
    #
    scenario_id2profile, dialogue_id2turns = read_evi_data(locale=args.locale)
    print(f'Read {len(scenario_id2profile)} profiles')
    print(f'Read {len(dialogue_id2turns)} dialogues')

    slot_order = [Slot.POSTCODE, Slot.NAME, Slot.DOB]
    policy_v = VerificationPolicy(
        nlu=build_nlu(name=args.nlu, locale=args.locale),
        verificator=build_model_v(args.model, locale=args.locale),
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
