""" Run models and evaluations for tasks

python projects/evi/eval_e.py \
    -eiv \
    --locale en_GB \
    --model naive

Copyright PolyAI Limited
"""

import argparse

import glog

from enrolment import EnrolmentEvaluator, EnrolmentPolicy, build_model_e
from nlu import build_nlu
from evi_dataset import DEFAULT_SLOT_ORDER, read_evi_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--locale', type=str,
        choices={"en_GB", "pl_PL", "fr_FR"},
        help='Locale to evaluate'
    )
    parser.add_argument(
        '--nlu', type=str,
        choices={"cautious", "seeking"},
        help='Which NLU model to evaluate'
    )
    parser.add_argument(
        '--model', type=str, default='0',
        choices={"0", "1", "2", "3"},
        help='Which model to evaluate'
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

    policy_e = EnrolmentPolicy(
        nlu=build_nlu(name=args.nlu, locale=args.locale),
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
    evaluator_e.print_report(draw_plots=args.plots)
    #
    print(args)


if __name__ == '__main__':
    _main()
    glog.info("Finished Enrolment Experiment!")
