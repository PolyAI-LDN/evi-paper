""" Run models and evaluations for tasks

python projects/evi/eval_e.py \
    -eiv \
    --locale en_GB \
    --model naive

Copyright PolyAI Limited
"""

import argparse

import glog

from nlu import build_nlu
from evi_dataset import DEFAULT_SLOT_ORDER, read_evi_data
from verification import (
    VerificationEvaluator, VerificationPolicy, build_model_v,
    randomized_verification_attempts
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


def _main():
    args = parse_args()
    slot_order = DEFAULT_SLOT_ORDER
    #
    scenario_id2profile, dialogue_id2turns = read_evi_data(locale=args.locale)
    print(f'Read {len(scenario_id2profile)} profiles')
    print(f'Read {len(dialogue_id2turns)} dialogues')

    dialogue_id2scenario_ids = randomized_verification_attempts(
        dialogue_id2turns=dialogue_id2turns,
        scenario_id2profile=scenario_id2profile,
        rate_genuine_to_impostor=1 / 1
    )

    policy_v = VerificationPolicy(
        nlu=build_nlu(name=args.nlu, locale=args.locale),
        verificator=build_model_v(args.model, locale=args.locale),
        slot_order=slot_order,
        hard_threshold=args.thresh,
    )
    evaluator_v = VerificationEvaluator()
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
    glog.info("Finished Verification Experiment!")
