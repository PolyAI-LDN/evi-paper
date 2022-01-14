""" Utils for normalising spelling

Copyright PolyAI Limited
"""
import re
from collections import OrderedDict
from functools import lru_cache
from typing import Dict, List

from num2words import num2words

CHAR2ALIASES = {
    "A": {"alfa", "alpha"},
    "B": {"bravo"},
    "C": {"charlie"},
    "D": {"delta"},
    "E": {"echo"},
    "F": {"foxtrot"},
    "G": {"golf"},
    "H": {"hotel"},
    "I": {"india"},
    "J": {"juliet", "juliette", 'juliett'},
    "K": {"kilo"},
    "L": {"lima"},
    "M": {"mike"},
    "N": {"november"},
    "O": {"oscar"},
    "P": {"papa"},
    "Q": {"quebec"},
    "R": {"romeo"},
    "S": {"sierra"},
    "T": {"tango"},
    "U": {"uniform"},
    "V": {"victor"},
    "W": {"whiskey"},
    "X": {"xray", "x-ray", "x ray"},
    "Y": {"yankee"},
    "Z": {"zulu"},
}

_LOCALE2FOR = {
    "en-GB": ['for', 'as in'],
    "pl-PL": ['jak'],
    "fr-FR": ['comme'],
}

_LOCALE2MULTI2ALIASES = {
    "en-GB": {
        2: ["double"],
        3: ["triple"],
    },
    "pl-PL": {
        2: ["podwójne"],
        3: ["potrójne"],
    },
    "fr-FR": {
        2: ["double"],
        3: ["triple"],
    },
}


@lru_cache()
def _get_num2aliases(locale: str = "en-GB") -> Dict[str, List[str]]:
    num2aliases = OrderedDict()
    # in descending order, longer to shorter lexicalisation
    for num in range(100, -1, -1):
        word = num2words(num, lang=locale)
        num2aliases[str(num)] = [word]
    num2aliases["0"] += ["oh", "nought"]
    return num2aliases


def _is_spelling(text: str) -> bool:
    return all(
        (c.isalpha() and c.isupper())
        or not c.isalpha()
        for c in text
    )


def normalise_spellings(text: str, locale: str = "en-GB") -> str:
    """ Preprocess text to normalise spellings and numbers

    Args:
        text: the original text
        locale: the language-region

    Returns:
        a text where any spelling substrings have been normalised
    """
    # preprocess
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(
        word.upper()
        if any(c.isupper() for c in word[1:])
        else word.lower()
        for word in text.split(" ")
    )
    # resolve "B for Bravo" -> "B"
    for_aliases = _LOCALE2FOR.get(locale, set())
    text = re.sub(
        r"\b([a-z]) " + f"({'|'.join(for_aliases)})" + r" \1\w+\b",
        lambda m: m.group(1).upper(),
        text
    )    # resolve "Bravo" -> "B"
    for char, aliases in CHAR2ALIASES.items():
        text = re.sub(
            r"\b" + f"({'|'.join(aliases)})" + r"\b",
            char.upper(),
            text,
        )
    # resolve "b" -> "B"
    text = re.sub(
        r"\b(\S)\b",
        lambda m: m.group(1).upper(),
        text
    )
    # resolve "one" -> "1"
    for num, aliases in _get_num2aliases(locale).items():
        text = re.sub(
            r"\b" + f"({'|'.join(aliases)})" + r"\b",
            num,
            text,
        )
    # resolve "double X" -> "XX"
    for factor, aliases in _LOCALE2MULTI2ALIASES.get(locale, {}).items():
        text = re.sub(
            r"\b" + f"({'|'.join(aliases)})" + r" (\S)",
            lambda m: m.group(2).upper() * factor,
            text,
        )
    # propagate upper case
    text = " ".join(
        word.upper()
        if any(c.isupper() or c.isnumeric() for c in word)
        else word
        for word in text.split(" ")
    )
    # remove spaces
    words = text.split(" ")
    text = ""
    for pos, word in enumerate(words):
        # lookahead
        try:
            next_word = words[pos + 1]
        except IndexError:
            text += word
            break
        text += word
        if not (_is_spelling(word) and _is_spelling(next_word)):
            text += " "
    return text
