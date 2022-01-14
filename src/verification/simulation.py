""" Run models and evaluations for tasks

python projects/evi/eval_e.py \
    -eiv \
    --locale en_GB \
    --model naive

Copyright PolyAI Limited
"""

from collections import defaultdict
from typing import Dict, List

import glog
from numpy.random import RandomState

from data_types import Profile, Turn


def randomized_verification_attempts(
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
