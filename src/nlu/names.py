""" NLU and utils for names

Copyright PolyAI Limited
"""
import re
from os import path
from typing import List, Optional, Set, Tuple

import names_dataset

from nlu.interface import AbstractParser
from nlu.nlu_utils import rm_duplicates, simplify_spelling


def _read_names(file_name: str) -> Set[str]:
    file_name = path.join(
        path.dirname(path.abspath(__file__)),
        "third_party", "census_data", file_name
    )
    out = set()
    with open(file_name) as f:
        for line in f:
            name, _ = line.split()[:2]
            name = name.upper()
            out.add(name)
    return out


def _read_census_data() -> Tuple[Set[str], Set[str]]:
    first_names = set()
    first_names = first_names.union(_read_names("dist.female.first"))
    first_names = first_names.union(_read_names("dist.male.first"))
    last_names = _read_names("dist.all.last")
    return first_names, last_names


class EviNameParser(AbstractParser):
    """ Trivial name NLU """

    def __init__(
        self,
        locale: str,
        strict: bool = False,
        use_nbest: bool = True
    ):
        """ Initialise

        Args:
            locale: the language-region locale
            strict: whether extract exact matches only
            use_nbest: whether extract from the ASR n-best
        """
        super().__init__(
            locale=locale,
            strict=strict,
            use_nbest=use_nbest
        )
        #
        self._db_v1 = names_dataset.NameDatasetV1()
        self._db_v2 = names_dataset.NameDataset()
        self._census_firsts, self._census_lasts = _read_census_data()
        #
        self._all_firsts = set()
        for t in self._db_v1.first_names:
            self._all_firsts.add(t.lower())
        for t in self._db_v2.first_names:
            self._all_firsts.add(t.lower())
        for t in self._census_firsts:
            self._all_firsts.add(t.lower())
        #
        self._all_lasts = set()
        for t in self._db_v1.last_names:
            self._all_lasts.add(t.lower())
        for t in self._db_v2.last_names:
            self._all_lasts.add(t.lower())
        for t in self._census_lasts:
            self._all_lasts.add(t.lower())

    def _is_first_name(self, text: str) -> bool:
        return text.lower() in self._all_firsts

    def _is_last_name(self, text: str) -> bool:
        return text.lower() in self._all_lasts

    def _is_full_name(self, text: str) -> bool:
        tokens = text.split(" ")
        if len(tokens) < 2:
            return False
        return all(
            self._is_first_name(t) or self._is_last_name(t)
            for t in tokens
        )

    def _split_full_text(self, text: str) -> List[Tuple[str, str]]:
        names = []
        tokens = text.split(" ")
        for i in range(len(tokens) - 1):
            first, last = tokens[i], tokens[i + 1]
            if not self._is_first_name(first):
                continue
            if self._strict and not self._is_full_name(f"{first} {last}"):
                continue
            names.append((first, last))
        return names

    def _split_full_spell(self, text: str) -> List[Tuple[str, str]]:
        text = text.replace(" ", "").upper()
        names = []
        for i in range(len(text) - 1, 0, -1):
            first, last = text[:i], text[i:]
            if not self._is_first_name(first):
                continue
            if self._strict and not self._is_full_name(f"{first} {last}"):
                continue
            names.append((first, last))
        return names

    def parse(  # noqa D003
        self,
        text: str,
        flags: Optional[List[str]] = None
    ) -> List[Tuple[str, str]]:
        flags = flags or []
        if "SPELL" in flags:
            return self._parse_spelling(text)
        else:
            return self._parse_value(text)

    def _parse_value(self, text: str) -> List[Tuple[str, str]]:
        if not self._is_full_name(text):
            return []
        names = []
        #
        # exact match
        for first, last in self._split_full_text(text):
            if first and last:
                names.append((first, last))
        #
        # search within context
        if not self._strict:
            tokens = text.split(" ")
            indices_first = []
            indices_last = []
            for i, t in enumerate(tokens):
                if self._is_first_name(t):
                    indices_first.append(i)
                if self._is_last_name(t):
                    indices_last.append(i)
            for i in indices_first:
                for j in indices_last:
                    if i == j:
                        continue
                    first = tokens[i]
                    last = tokens[j]
                    names.append((first, last))
        names = rm_duplicates(names)
        return names

    def _parse_spelling(self, text: str) -> List[Tuple[str, str]]:
        text = simplify_spelling(text)
        #
        # exact match
        names = []
        for first, last in self._split_full_spell(text):
            if first and last:
                names.append((first, last))
        #
        # search within context
        if not self._strict:
            for span in re.findall(r"\b(\w+)\b", text):
                if not span.isupper():
                    continue
                for first, last in self._split_full_spell(span):
                    if first and last:
                        names.append((first, last))
        return names

    def demo(self):
        """ Launch an interactive demo """
        while True:
            text = input(">")
            outs = self.parse(text, [])
            for i, o in enumerate(outs):
                print(f'Spelling {i}: {o}')
            outs = self.parse(text, ['SPELL'])
            for i, o in enumerate(outs):
                print(f'Value    {i}: {o}')


# m = EviNameParser(locale="en-GB", strict=True)._parse_spelling("JOHNLENNON")
# print(m)
#
# exit()

if __name__ == '__main__':
    print("Done!")
