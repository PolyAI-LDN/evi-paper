""" Models for NLU

Copyright PolyAI Limited
"""
from data_types import Slot, Turn
from nlu.dates import EviDateValueExtractor
from nlu.names import EviNameValueExtractor
from nlu.postcodes import EviPostcodeValueExtractor
from nlu.types_ import NluOutput


class Nlu(object):
    """ Abstract model for identification """

    def __init__(
        self,
        postcode_extractor: EviPostcodeValueExtractor,
        name_extractor: EviNameValueExtractor,
        date_extractor: EviDateValueExtractor,
    ):
        """ Initialise

        Args:
            postcode_extractor: a PostcodeParser
            name_extractor: a NameParser
            date_extractor: a DateParser
        """
        self._postcode_extractor = postcode_extractor
        self._name_extractor = name_extractor
        self._date_extractor = date_extractor

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

        if request_postcode and self._postcode_extractor is not None:
            postcodes.extend(
                self._postcode_extractor.extract_nbest(texts=turn.nbest)
            )
        if request_name:
            names.extend(
                self._name_extractor.extract_nbest(
                    texts=turn.nbest,
                    flags=['SPELL'] if turn.requested_spelling else []
                )
            )
        if request_dob:
            dobs.extend(
                self._date_extractor.extract_nbest(texts=turn.nbest)
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
