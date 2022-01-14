""" Data readers and types

Copyright PolyAI Limited
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
from ast import literal_eval
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import glog

_DATA_DIR = "../tmp/evi"


class Slot:
    """ Enumeration of turn slots """
    POSTCODE = "postcode"
    NAME = "name"
    DOB = "dob"


@dataclass()
class Profile(object):
    """ An entry in the profiles KB """
    scenario_id: str
    postcode: str
    name_first: str
    name_last: str
    dob_str: str  # date in iso format YYYY-MM-DD

    def has(self, item: str) -> bool:
        """ Whether a profile's slot is populated """
        if item == Slot.NAME:
            return bool(self.name_full)
        elif item == Slot.DOB:
            return bool(self.dob_str)
        elif item == Slot.POSTCODE:
            return bool(self.postcode)
        else:
            raise ValueError(f"Unknown item {item}")

    def get_item_hash(self, item: str) -> str:
        """ Get a hash of a slot - useful for comparisons """
        if item == Slot.NAME:
            return self.name_full.upper()
        elif item == Slot.DOB:
            return self.dob_str
        elif item == Slot.POSTCODE:
            return self.postcode.replace(' ', '').replace('-', '').upper()
        else:
            raise ValueError(f"Unknown item {item}")

    @property
    def dob(self) -> Optional[dt.date]:
        """ The date of birth (as date) """
        try:
            return dt.date.fromisoformat(self.dob_str)
        except ValueError:
            return None

    @property
    def name_full(self) -> str:
        """ The full name of the user """
        parts = []
        if self.name_first:
            parts.append(self.name_first)
        if self.name_last:
            parts.append(self.name_last)
        return ' '.join(parts)


@dataclass()
class Turn(object):
    """ User input of a dialogue turn """
    dialogue_id: str
    speaker_id: str
    scenario_id: str
    turn_num: int
    item: str
    attempt_num: int
    transcription: str
    nbest: List[str]
    prime_letter: str
    prime_month: str

    @property
    def requested_spelling(self):
        """Whether we have explicitly requested the user to spell"""
        return self.item == Slot.NAME and self.attempt_num == 2


def read_dialogues(
    input_file: str,
) -> Dict[str, List[Turn]]:
    """ Read dialogue data """
    is_viasat = 'viasat' in input_file
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
            if is_viasat:
                # skip viasat non-item collection turns
                if dictrow.get('item', 'N/A') == 'start':
                    continue
                if any(
                    t.lower() in {
                        'yes', 'no', 'yeah', 'correct', "that's correct"
                    } for t in nbest
                ):
                    continue
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


def read_profiles(
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


def filter_turns(
    turns: List[Turn],
    item: str,
    max_attempts: int
) -> List[Turn]:
    """ Filter turns by item

    Args:
        turns: a list of turns
        item: the item to filter by
        max_attempts: max number of times to ask user about each slot

    Returns:
        a list of filtered turns
    """
    selected = []
    for t in turns:
        if t.item == item:
            selected.append(t)
    return selected[:max_attempts]


def _main():
    """ demo """
    is_viasat = True
    lang_code = 'en'
    #
    if is_viasat:
        profiles_file = os.path.join(
            _DATA_DIR, 'viasat',
            f"viasat.records.{lang_code}.csv"
        )
        dialogues_file = os.path.join(
            _DATA_DIR, 'viasat',
            f"viasat.dialogues.{lang_code}.tsv"
        )
    else:
        profiles_file = os.path.join(
            _DATA_DIR, 'dataset',
            f"records.{lang_code}.csv"
        )
        dialogues_file = os.path.join(
            _DATA_DIR, 'dataset',
            f"dialogues.{lang_code}.tsv"
        )
    #
    scenario_id2profile = read_profiles(profiles_file)
    print(f'Loaded {len(scenario_id2profile)} profiles')
    #
    dialogue_id2turns = read_dialogues(dialogues_file)
    print(f'Loaded {len(dialogue_id2turns)} dialogues')


if __name__ == '__main__':
    _main()
    print("Done!")
