""" Data readers

Copyright PolyAI Limited
"""
from __future__ import annotations

import csv
import json
import os
from ast import literal_eval
from collections import defaultdict
from typing import Dict, List, Optional

import glog

from data_types import Profile, Turn


_DATA_DIR = "../data"


def _read_dialogues(
    input_file: str,
) -> Dict[str, List[Turn]]:
    """ Read dialogue data """
    dialogue_id2turns = defaultdict(list)
    n_multiple_targets = 0
    n_no_targets = 0
    with open(input_file, 'r') as fin:
        reader = csv.DictReader(fin, delimiter='\t')
        for dictrow in reader:
            #
            try:
                nbest = [x for x in json.loads(dictrow['nbest']) if x]
            except json.decoder.JSONDecodeError:
                nbest = literal_eval(dictrow.get('nbest', '[]'))
            dialogue_id = dictrow['dialogue_id']
            scenario_id = dictrow['scenario_id']
            if scenario_id.startswith('['):
                _targets_incl_none = (literal_eval(scenario_id) + [None])
                scenario_id = _targets_incl_none[0]
                if dialogue_id not in dialogue_id2turns:
                    if _targets_incl_none[0] is None:
                        n_no_targets += 1
                    elif len(_targets_incl_none[:-1]) > 1:
                        n_multiple_targets += 1
            t = Turn(
                dialogue_id=dialogue_id,
                speaker_id=dictrow.get('speaker_id', 'N/A'),
                scenario_id=scenario_id,
                turn_num=int(dictrow['turn_num']),
                item=dictrow.get('item', 'N/A'),
                attempt_num=int(dictrow['attempt_num']),
                transcription=dictrow['transcription'],
                nbest=nbest,
                prime_letter=dictrow.get('prime_letter', 'N/A'),
                prime_month=dictrow.get('prime_month', 'N/A')
            )
            dialogue_id2turns[t.dialogue_id].append(t)
    dialogue_id2turns = {
        idx: sorted(turns, key=lambda t: t.turn_num)
        for idx, turns in dialogue_id2turns.items()
    }
    glog.info(f'Loaded {len(dialogue_id2turns)} dialogues')
    glog.warn(
        f'{n_no_targets} dialogues with NO targets (set as None)'
    )
    glog.warn(
        f'{n_multiple_targets} dialogues with MULTIPLE targets (set to 1st)'
    )
    return dialogue_id2turns


def _read_profiles(
    input_file: str,
    max_n_profiles: Optional[int] = None
) -> Dict[str, Profile]:
    """ Read KB profile data

    Args:
        input_file:
        max_n_profiles: maximum number of profiles to read

    Returns:
    """
    scenario_id2profile = {}
    with open(input_file, 'r') as fin:
        reader = csv.DictReader(fin, delimiter=',')
        for dictrow in reader:
            if (
                max_n_profiles is not None
                and len(scenario_id2profile) >= max_n_profiles
            ):
                break
            p = Profile(
                scenario_id=dictrow['scenario_id'],
                postcode=dictrow['postcode'],
                name_first=dictrow['name_first'],
                name_last=dictrow['name_last'],
                dob_str=dictrow['dob'],
            )
            scenario_id2profile[p.scenario_id] = p
    glog.info(f'Loaded {len(scenario_id2profile)} profiles')
    return scenario_id2profile


def read_evi_data(locale: str):
    locale = locale.replace("_", "-")
    lang_code = locale.split('-')[0]
    profiles_file = os.path.join(_DATA_DIR, f"records.{lang_code}.csv")
    dialogues_file = os.path.join(_DATA_DIR, f"dialogues.{lang_code}.tsv")
    scenario_id2profile = _read_profiles(profiles_file)
    dialogue_id2turns = _read_dialogues(dialogues_file)
    return scenario_id2profile, dialogue_id2turns


def _main():
    """ demo """
    lang_code = 'en'
    #
    profiles_file = os.path.join(_DATA_DIR, f"records.{lang_code}.csv")
    dialogues_file = os.path.join(_DATA_DIR, f"dialogues.{lang_code}.tsv")
    #
    scenario_id2profile = _read_profiles(profiles_file)
    print(f'Loaded {len(scenario_id2profile)} profiles')
    #
    dialogue_id2turns = _read_dialogues(dialogues_file)
    print(f'Loaded {len(dialogue_id2turns)} dialogues')


if __name__ == '__main__':
    _main()
    print("Done!")
