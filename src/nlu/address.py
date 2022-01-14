""" NLU and utils for address

Copyright PolyAI Limited
"""
import abc
import re
from typing import List, Tuple

from address_parser import Parser


class AddressParser(abc.ABC):
    """ Address NLU """

    def __init__(self, use_nbest: bool = False):
        """ Initialise

        Args:
            use_nbest: whether to extract from ASR n-best list
        """
        self._use_nbest = use_nbest

    @classmethod
    def create(cls, name: str):
        """ Create a new enrolment model """
        if name == 'trivial':
            return TrivialAddressParser(
                use_nbest=False
            )
        elif name == 'simple':
            return SimpleAddressParser(
                use_nbest=True
            )
        raise NotImplementedError(f'No Address Parser {name}')

    @abc.abstractmethod
    def parse(self, text: str) -> List[Tuple[str, str, str]]:
        """ Parse text into address """
        raise NotImplementedError()

    def parse_nbest(self, texts: List[str]) -> List[Tuple[str, str, str]]:
        """ Parse a list of texts into addresses """
        if not self._use_nbest:
            texts = texts[:1]
        addresses = []
        for text in texts:
            try:
                ps = self.parse(text)
            except Exception:
                ps = [(text, '', '')]
            addresses.extend(ps)
        return addresses


class TrivialAddressParser(AddressParser):
    """ Trivial address NLU """

    def parse(self, text: str) -> List[Tuple[str, str, str]]:  # noqa D003
        return [(text, '', '')]


class SimpleAddressParser(AddressParser):
    """ Simple address NLU """

    def __init__(self, use_nbest: bool = False):
        """Initialise"""
        super().__init__(use_nbest=use_nbest)
        self._parser = Parser()

    def parse(self, text: str) -> List[Tuple[str, str, str]]:  # noqa D003
        p = self._parser.parse(text)
        num = ''
        try:
            num = p.number.tnumber
        except Exception:
            pass
        street = []
        try:
            if p.road.name:
                street.append(p.road.name)
            if p.road.suffix:
                street.append(p.road.suffix)
        except Exception:
            pass
        street = ' '.join(street)
        floa = ''
        if street or num:
            floa = text
        #
        street = street.lower().strip()
        num = num.lower().strip()
        street = street if street != 'none' else ''
        num = num if num != 'none' else ''
        street = street or ''
        num = num or ''
        street = re.sub(r"\bave\b", "avenue", street)
        street = re.sub(r"\brdg\b", "ridge", street)
        street = re.sub(r"\bcrk\b", "creek", street)
        street = re.sub(r"\bpk\b", "pike", street)
        street = re.sub(r"\bmtn\b", "mountain", street)
        street = re.sub(r"\bdr\b", "drive", street)
        street = re.sub(r"\bvly\b", "valley", street)
        street = re.sub(r"\bky\b", "key", street)
        street = re.sub(r"\bst\b", "street", street)
        street = re.sub(r"\bln\b", "lane", street)
        street = re.sub(r"\bpl\b", "place", street)
        street = re.sub(r"\bbr\b", "branch", street)
        street = re.sub(r"\brd\b", "road", street)
        street = re.sub(r"\bhl\b", "hill", street)
        street = re.sub(r"\bhls\b", "hills", street)
        street = re.sub(r"\bblvd\b", "boulevard", street)
        street = re.sub(r"\bhwy\b", "highway", street)
        street = re.sub(r"\blk\b", "lake", street)
        street = re.sub(r"\bcir\b", "circle", street)
        street = re.sub(r"\bvw\b", "view", street)
        street = re.sub(r"\btrl\b", "trail", street)
        street = re.sub(r"\bcv\b", "cove", street)
        street = re.sub(r"\bholw\b", "hollow", street)
        street = re.sub(r"\bpnes\b", "pines", street)
        street = re.sub(r"\bn\b", "north", street)
        street = re.sub(r"\bs\b", "south", street)
        street = re.sub(r"\be\b", "east", street)
        street = re.sub(r"\bw\b", "west", street)
        street = re.sub(r"\bne\b", "north east", street)
        street = re.sub(r"\bnw\b", "north west", street)
        street = re.sub(r"\bse\b", "south east", street)
        street = re.sub(r"\bsw\b", "south west", street)
        return [(floa, num, street)]
