""" Models for NLU

Copyright PolyAI Limited
"""
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Tuple

from nlu.dates import EviDateParser
from nlu.names import EviNameParser
from nlu.postcodes import EviPostcodeParser
from readers import Slot, Turn


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


class Nlu(object):
    """ Abstract model for identification """

    def __init__(
        self,
        postcode_parser: EviPostcodeParser,
        name_parser: EviNameParser,
        date_parser: EviDateParser,
    ):
        """ Initialise

        Args:
            postcode_parser: a PostcodeParser
            name_parser: a NameParser
            date_parser: a DateParser
        """
        self._postcode_parser = postcode_parser
        self._name_parser = name_parser
        self._date_parser = date_parser

    def run_nlu(
        self,
        turn: Turn,
        request_postcode: bool = False,
        request_name: bool = False,
        request_dob: bool = False,
    ) -> NluOutput:
        """Execute NLU on a turn

        Args:
            turn: a dialogue turn
            request_postcode: whether to parse postcodes
            request_name: whether to parse names
            request_dob: whether to parse DOBs

        Returns:
            an NluOutput
        """
        postcodes = []
        names = []
        dobs = []

        if request_postcode and self._postcode_parser is not None:
            postcodes.extend(
                self._postcode_parser.parse_nbest(texts=turn.nbest)
            )
        if request_name:
            names.extend(
                self._name_parser.parse_nbest(
                    texts=turn.nbest,
                    flags=['SPELL'] if turn.requested_spelling else []
                )
            )
        if request_dob:
            dobs.extend(
                self._date_parser.parse_nbest(texts=turn.nbest)
            )

        out = NluOutput(
            texts=turn.nbest,
            postcodes=postcodes,
            names=names,
            dobs=dobs,
        )
        return out

    def run_nlu_for_item(
        self,
        turn: Turn,
        item: str
    ) -> NluOutput:
        """Execute NLU on a turn for a particular item

        Args:
            turn: a dialogue turn
            item: the specified item

        Returns:
            an NluOutput
        """
        if item in Slot.POSTCODE:
            return self.run_nlu(turn=turn, request_postcode=True)
        elif item == Slot.NAME:
            return self.run_nlu(turn=turn, request_name=True)
        elif item == Slot.DOB:
            return self.run_nlu(turn=turn, request_dob=True)
        else:
            raise ValueError(f"Unknown item {item}")
