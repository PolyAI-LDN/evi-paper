""" NLU data types

Copyright PolyAI Limited
"""
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class NluOutput(object):
    """Dataclass for NLU outputs"""
    texts: List[str] = field(default_factory=lambda: [])
    postcodes: List[str] = field(default_factory=lambda: [])
    names: List[Tuple[str, str]] = field(default_factory=lambda: [])
    dobs: List[dt.date] = field(default_factory=lambda: [])

    def get_first_names(self) -> List[str]:
        """ a list of extracted first names """
        first_names = []
        for first, _ in self.names:
            if not first:
                continue
            first_names.append(first)
        return first_names

    def get_last_names(self) -> List[str]:
        """ a list of extracted last names """
        last_names = []
        for _, last in self.names:
            if not last:
                continue
            last_names.append(last)
        return last_names

    def get_full_names(self) -> List[str]:
        """ a list of extracted full names """
        full_names = []
        for first, last in self.names:
            if not first:
                continue
            if not last:
                continue
            full_names.append(f"{first} {last}")
        return full_names
