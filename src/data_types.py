""" Data types

Copyright PolyAI Limited
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional


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