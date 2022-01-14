""" Run models and evaluations for tasks

python projects/evi/eval_e.py \
    -eiv \
    --locale en_GB \
    --model naive

Copyright PolyAI Limited
"""

import argparse

import glog

from identification import (
    IdentificationEvaluator, IdentificationPolicy, ProfileDatastore,
    build_model_i
)
from nlu import build_nlu
from evi_dataset import DEFAULT_SLOT_ORDER, load_evi_data


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
        '--model', type=str,
        choices={
            "oracle", "none",
            "exact-1", "exact-0.5",
            "fuzzy-1", "fuzzy-0.5",
        },
        help='Which model to evaluate'
    )
    parser.add_argument(
        '--kbo',
        action='store_true',
        help='Whether to use the KB oracle for identification'
    )
    parser.add_argument(
        '--eager',
        action='store_true',
        help='Whether to terminate identification asap'
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
    scenario_id2profile, dialogue_id2turns = load_evi_data(locale=args.locale)
    print(f'Read {len(scenario_id2profile)} profiles')
    print(f'Read {len(dialogue_id2turns)} dialogues')

    policy_i = IdentificationPolicy(
        db=ProfileDatastore(
            scenario_id2profile,
            searchable_by_postcode=True,
            searchable_by_name=False,
            searchable_by_dob=False,
            return_all_profiles=False,
        ),
        nlu=build_nlu(name=args.nlu, locale=args.locale),
        identificator=build_model_i(name=args.model, locale=args.locale),
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
    print(args)


if __name__ == '__main__':
    _main()
    glog.info("Finished Identification Experiment!")
