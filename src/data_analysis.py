""" Analyse data to extract descriptive statistics

python projects/evi/data_analysis.py --locale en_GB

--email optional flag to perform email analysis instead

Copyright PolyAI Limited
"""

import argparse
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from nlu.names import EviNameParser
from readers import Profile, Turn, read_dialogues, read_profiles

_DATA_DIR = "../tmp/evi"


@dataclass()
class _SankeyLink:
    from_: int
    to: int
    value: float
    colour: str
    label: str = ''


def _get_colour(from_: str, to: str, prime: str):
    if to == 'other':
        return _grey(0.5)
    if from_ != to and not (from_ == 'start' and prime == to):
        alpha = 0.8
    else:
        # weaker if not an interesting transition
        alpha = 0.2
    if to == prime:  # in {'nato', 'word', True}:
        return _red(alpha)
    else:
        return _blue(alpha)


def _get_word_january(locale: str):
    lang = locale.split("-")[0]
    if lang == "en":
        return "January"
    elif lang == "pl":
        return "styczeń"
    elif lang == "fr":
        return "janvier"
    raise NotImplementedError(f"Unk `january' name for {lang}")


def _get_word_one(locale: str):
    lang = locale.split("-")[0]
    if lang == "en":
        return "1"
    elif lang == "pl":
        return "1"
    elif lang == "fr":
        return "1"
    raise NotImplementedError(f"Unk `one' name for {lang}")


def _grey(alpha=1.0):
    return f'rgba(211, 211, 211, {alpha:.2f})'


def _red(alpha=1.0):
    return f'rgba(255, 100, 100, {alpha:.2f})'


def _blue(alpha=1.0):
    return f'rgba(100, 156, 255, {alpha:.2f})'


class ProfileAnalyser(object):
    """ Statistical analyser of profile data """

    def __init__(self):
        """ Initialise """
        self.profiles = []
        self.counter_first = Counter()
        self.counter_last = Counter()
        self.counter_full = Counter()
        self.counter_dob = Counter()
        self.counter_postcode = Counter()
        self.counter_email = Counter()

    def increment(self, p: Profile):
        """ Add a new profile to the analyser """
        self.profiles.append(p)
        self.counter_first[p.name_first] += 1
        self.counter_last[p.name_last] += 1
        self.counter_full[p.name_full] += 1
        self.counter_dob[p.dob_str] += 1
        self.counter_postcode[p.postcode] += 1

        if p.email:
            self.counter_email[p.email] += 1

    def print_profile_analysis(self):
        """ Print summary of profile data analysis """
        n_profiles = len(self.profiles)
        print('=================')
        print('Profiles Analysis')
        print('=================')
        print(f'#profiles            : {n_profiles}')
        print(f'#unique postcodes    : {len(self.counter_postcode.keys())}')
        print(f'#unique names (first): {len(self.counter_first.keys())}')
        print(f'#unique names (last) : {len(self.counter_last.keys())}')
        print(f'#unique names (full) : {len(self.counter_full.keys())}')
        print(f'#unique dobs         : {len(self.counter_dob.keys())}')
        print(f'#unique emails       : {len(self.counter_email.keys())}')


class NameParserAnalyser(object):
    """ Analyse max percentage of full names that can be retrieved """

    def __init__(self, parser: EviNameParser):
        """ Initialise

        Args:
            parser: the name parser
        """
        self.parser = parser
        self._failed_fulls = set()
        self._failed_firsts = set()
        self._failed_lasts = set()
        self._n_seen = 0
        self._n_correct = 0

    def increment(self, profile: Profile):
        """ Add a new profile to the analyser """
        self._n_seen += 1
        text = profile.name_full.lower()
        names = self.parser.parse_nbest([text])
        parsed_correctly = False
        if names:
            first, last = names[0]
            if (
                first.lower() == profile.name_first.lower()
                and last.lower() == profile.name_last.lower()
            ):
                parsed_correctly = True
        if parsed_correctly:
            self._n_correct += 1
        else:
            self._failed_fulls.add(profile.name_full)
            self._failed_firsts.add(profile.name_first)
            self._failed_lasts.add(profile.name_last)

    def print_parsing_analysis(self):
        """ Print summary of parser analysis on profiles """
        print("=======Name Parser Analysis=======")
        coverage = self._n_correct / self._n_seen
        print("-------  Unparseable Full   -------")
        print(sorted(list(self._failed_fulls)))
        print("-------  Unparseable First  -------")
        print(sorted(list(self._failed_firsts)))
        print("-------  Unparseable Last   -------")
        print(sorted(list(self._failed_lasts)))
        print("-----------------------")
        print(f'#db(names):  {self._n_seen}')
        print(f'#unparsable: {self._n_seen - self._n_correct}')
        print(f'%coverage:   {coverage * 100.0:.2f}')


_LOCALE2NATO = {
    "en-GB": {
        "alfa", "alpha",
        "bravo",
        "charlie",
        "delta", "echo",
        "foxtrot", "golf", "hotel", "india",
        "juliet", "juliette", 'juliett',
        "kilo", "lima",
        "mike",
        "november",
        "oscar",
        "papa", "quebec",
        "romeo",
        "sierra", "tango", "uniform",
        "victor",
        "whiskey",
        "xray", "x-ray", "x ray",
        "yankee", "zulu",
    },
    "pl-PL": set(),
    "fr-FR": set()
}

_LOCALE2FOR_NATO = {
    "en-GB": {"as in", "for"},
    "pl-PL": {'jak'},
    "fr-FR": {'comme'},
}

_LOCALE2MONTHS = {
    "en-GB": {
        "january", "february", "march",
        "april", "may", "june",
        "july", "august", "september",
        "october", "november", "december"
    },
    "pl-PL": {
        # nominative
        "styczeń", "luty",
        "marzec", "kwiecień", "maj",
        "czerwiec", "lipiec", "sierpień",
        "wrzesień", "październik", "listopad",
        "grudzień",
        # genitive
        "stycznia", "lutego",
        "marca", "kwietnia", "maja",
        "czerwca", "lipca", "sierpnia",
        "września", "października", "listopada",
        "grudnia",
        # english
        "january", "february", "march",
        "april", "may", "june",
        "july", "august", "september",
        "october", "november", "december"
    },
    "fr-FR": {
        "janvier", "février",
        "mars", "avril", "mai",
        "juin", "juillet", "août",
        "septembre", "octobre", "novembre",
        "décembre",
    }
}


def _has_spelling(turn: Turn) -> bool:
    """ Heuristic to check whether a turn uses char by char spelling """
    threshold = 5
    for t in turn.nbest:
        no_short_tokens = 0
        for token in t.split(" "):
            if len(token) == 1:
                no_short_tokens += 1
            if no_short_tokens > threshold:
                return True
    return False


def _to_case_encoding(text: str) -> str:
    """ This Is anExample -> Xxxx Xxx xxXxxxxx """
    transformed = ''
    for c in text:
        if c.islower():
            transformed += 'x'
        elif c.isupper():
            transformed += 'X'
        else:
            transformed += ' '
    return transformed


def _has_nonstandard_nato(
    text: str,
    min_tokens=4,
    min_unique_tokens=4
) -> bool:
    pattern_case = r'(\b(?:Xx{2,})(?: Xx{2,})+\b)'
    # check for Camel Case pattern Xxx Xxxxx Xxxx
    m = re.search(pattern_case, _to_case_encoding(text))
    if not m:
        return False
    start, end = m.span()
    span = text[start:end]
    tokens = span.split(" ")
    # is it long enough?
    if len(tokens) < min_tokens:
        return False
    # does it have a diverse vocabulary?
    vocab = set(tokens)
    vocab -= {"Prix"}  # hack to exclude samples
    if len(vocab) < min_unique_tokens:
        return False
    return True


class DialogueAnalyser(object):
    """ Statistical analyser for dialogue data """

    def __init__(
        self,
        locale: str,
        scenario_id2profile: Optional[Dict[str, Profile]] = None,
        email_mode=False,
    ):
        """ Initialise

        Args:
            locale: the locale to evaluate in (en-GB, fr-FR, pl-PL)
            scenario_id2profile: (if known) the dict of profiles
                                 indexed by scenario_id
            email_mode: whether to evaluate emails
        """
        self.locale = locale
        self.scenario_id2profile = scenario_id2profile or {}
        self.dialogues = []
        self.speakers = set()
        self.scenarios = set()
        self.nbest_lengths = defaultdict(list)
        self.email_mode = email_mode

        if self.email_mode:
            self.prime_email = Counter()
        else:
            self.prime_postcode = Counter()
            self.prime_name = Counter()
            self.prime_dob = Counter()
            self.spelling_transitions = Counter()
            self.month_transitions = Counter()

    def _get_spelling_behaviour(self, turn: Turn) -> str:
        if self._has_nato(turn):
            utterance_type = "nato"
        elif not turn.nbest:
            utterance_type = "other"
        else:
            utterance_type = "letter"
        return utterance_type

    def _get_date_behaviour(self, turn: Turn) -> str:
        if self._has_month(turn):
            utterance_type = "word"
        elif not turn.nbest:
            utterance_type = "other"
        else:
            utterance_type = "digit"
        return utterance_type

    def increment(self, turns: List[Turn]):
        """ Add a new dialogue to the analyser """
        self.dialogues.append(turns)
        t = turns[0]
        self.speakers.add(t.speaker_id)
        self.scenarios.add(t.scenario_id)

        if self.email_mode:
            for i in range(3):
                utterance_type = "word"
                if _has_spelling(turns[i]):
                    utterance_type = "letter"
                elif self._has_nato(turns[i]):
                    utterance_type = "nato"
                self.prime_email[(i, t.prime_letter, utterance_type)] += 1

        else:
            for i in range(3):
                self.prime_postcode[
                    (i, t.prime_letter, self._has_nato(turns[i]))
                ] += 1
            for i in range(3):
                self.prime_name[
                    (i, t.prime_letter, self._has_nato(turns[3 + i]))
                ] += 1
            for i in range(3):
                self.prime_dob[
                    (i, t.prime_month, self._has_month(turns[6 + i]))
                ] += 1
            #
            # track behaviours for communicating spelling
            for i in range(1, 3):
                # postcode spelling turns
                self.spelling_transitions[
                    (
                        i - 1,  # turn at start of transition
                        self._get_spelling_behaviour(turns[i - 1]),
                        self._get_spelling_behaviour(turns[i]),
                        (t.prime_letter if i >= 2 else 'letter')  # priming
                    )
                ] += 1
            # name spelling turn
            self.spelling_transitions[
                (
                    2,
                    self._get_spelling_behaviour(turns[2]),
                    self._get_spelling_behaviour(turns[5]),
                    t.prime_letter
                )
            ] += 1
            #
            # track behaviours for communicating date
            self.month_transitions[
                (
                    5,  # turn at start of transition
                    t.prime_month,
                    self._get_date_behaviour(turns[6]),
                    t.prime_month
                )
            ] += 1
            for i in range(7, 9):
                self.month_transitions[
                    (
                        i - 1,  # turn at start of transition
                        self._get_date_behaviour(turns[i - 1]),
                        self._get_date_behaviour(turns[i]),
                        t.prime_month
                    )
                ] += 1

        for i in range(self._get_dialogue_length()):
            self.nbest_lengths[i].append(len(turns[i].nbest))

    def _get_dialogue_length(self):
        return 3 if self.email_mode else 9

    def _has_nato(self, turn: Turn) -> bool:
        nato_for = _LOCALE2FOR_NATO[self.locale]
        nato_words = _LOCALE2NATO[self.locale]
        try:
            p = self.scenario_id2profile[turn.scenario_id]
            nato_words -= {p.name_first.lower(), p.name_last.lower()}
        except KeyError:
            pass
        #
        pattern_bravo = (
            rf'\b({"|".join(nato_words)})\b'
            if nato_words else r'$x'  # i.e. a regex that never matches
        )
        #
        pattern_for = rf'\b({"|".join(nato_for)})\b' if nato_for else r'$x'
        #
        return (
            # check whether (B for) *Bravo*, etc. is mentioned
            any(
                len(t.split(" ")) >= 3 and
                len(re.findall(pattern_bravo, t, re.I)) >= 1
                for t in turn.nbest
            )
            # check whether B *for* Bravo, etc. is mentioned
            or any(
                len(re.findall(pattern_for, t, re.I)) >= 2
                for t in turn.nbest
            )
            or any(
                _has_nonstandard_nato(t)
                for t in turn.nbest
            )
        )

    def _has_month(self, turn: Turn) -> bool:
        month_words = _LOCALE2MONTHS[self.locale]
        return any(
            any(w in t.lower() for w in month_words)
            for t in turn.nbest
        )

    def print_dialogue_analysis(self):
        """ Print summary of dialogue data analysis """
        n_dialogues = len(self.dialogues)
        print('=================')
        print('Dialogue Analysis')
        print('=================')
        print(f'#unique dialogues: {n_dialogues}')
        print(f'#unique speakers : {len(self.speakers)}')
        print(f'#unique scenarios: {len(self.scenarios)}')

    def print_asr_analysis(self):
        """ Print summary of dialogue data analysis wrt ASR """
        print('============================================')
        print('               ASR Analysis                 ')
        print('============================================')
        for i in range(self._get_dialogue_length()):
            print('--------')
            print(f'Turn {i + 1}')
            print('--------')
            n_empty = sum(1 for x in self.nbest_lengths[i] if x == 0)
            avg = np.mean(self.nbest_lengths[i])
            std = np.std(self.nbest_lengths[i])
            _min = np.min(self.nbest_lengths[i])
            med = np.median(self.nbest_lengths[i])
            _max = np.max(self.nbest_lengths[i])
            print(f'nbest length avg : {avg:.2f}')
            print(f'nbest length std : {std:.2f}')
            print(f'nbest length min : {_min:.2f}')
            print(f'nbest length med : {med:.2f}')
            print(f'nbest length max : {_max:.2f}')
            print(f'#no transcription: {n_empty}')

    def print_email_priming(self):
        """ Print priming analysis concerning emails"""
        print('********************************************')
        print('*************** Email priming **************')
        print('********************************************')
        for i in range(3):
            print('--------------------------------------------')
            print(f'-                Attempt {i + 1}                 -')
            print('--------------------------------------------')
            nato_primed_nato = self.prime_email[(i, 'nato', 'nato')]
            nato_primed_letter = self.prime_email[(i, 'nato', 'letter')]
            nato_primed_word = self.prime_email[(i, 'nato', 'word')]
            #
            letter_primed_nato = self.prime_email[(i, 'letter', 'nato')]
            letter_primed_letter = self.prime_email[(i, 'letter', 'letter')]
            letter_primed_word = self.prime_email[(i, 'letter', 'word')]
            #
            unprimed_nato = self.prime_email[(i, 'word', 'nato')]
            unprimed_letter = self.prime_email[(i, 'word', 'letter')]
            unprimed_word = self.prime_email[(i, 'word', 'word')]
            print('Hear\\Say| letter\tnato\t\tword\t\t|total')
            if i == 2:
                print('--------------------------------------------')
                a = letter_primed_letter
                b = letter_primed_nato
                c = letter_primed_word
                pa = a / (a + b + c) * 100
                pb = b / (a + b + c) * 100
                pc = c / (a + b + c) * 100
                print(f'  letter| {a:3d}({pa:2.0f}%)\t{b:3d}({pb:2.0f}%)\t{c:3d}({pc:2.0f}%)\t|{a + b + c}')  # noqa
                a = nato_primed_letter
                b = nato_primed_nato
                c = nato_primed_word
                pa = a / (a + b + c) * 100
                pb = b / (a + b + c) * 100
                pc = c / (a + b + c) * 100
                print(f'    nato| {a:3d}({pa:2.0f}%)\t{b:3d}({pb:2.0f}%)\t{c:3d}({pc:2.0f}%)\t|{a + b + c}')  # noqa
                a = unprimed_letter
                b = unprimed_nato
                c = unprimed_word
                pa = a / (a + b + c) * 100
                pb = b / (a + b + c) * 100
                pc = c / (a + b + c) * 100
                print(f'    word| {a:3d}({pa:2.0f}%)\t{b:3d}({pb:2.0f}%)\t{c:3d}({pc:2.0f}%)\t|{a + b + c}')  # noqa
            a = unprimed_letter + letter_primed_letter + nato_primed_letter
            b = unprimed_nato + letter_primed_nato + nato_primed_nato
            c = unprimed_word + letter_primed_word + nato_primed_word
            pa = a / (a + b + c) * 100
            pb = b / (a + b + c) * 100
            pc = c / (a + b + c) * 100
            print('--------------------------------------------')
            print(f'   total| {a:3d}({pa:2.0f}%)\t{b:3d}({pb:2.0f}%)\t{c:3d}({pc:2.0f}%)\t|{a + b + c}')  # noqa

    def print_postcode_priming(self):
        """ Print priming analysis concerning postcodes"""
        print('********************************************')
        print('************* Postcode priming *************')
        print('********************************************')
        for i in range(3):
            print('--------------------------------------------')
            print(f'-                Attempt {i + 1}                 -')
            print('--------------------------------------------')
            primed_nato = self.prime_postcode[(i, 'nato', True)]
            primed_letter = self.prime_postcode[(i, 'nato', False)]
            unprimed_nato = self.prime_postcode[(i, 'letter', True)]
            unprimed_letter = self.prime_postcode[(i, 'letter', False)]
            print('Hear\\Say| letter\tnato\t\t|total')
            if i == 2:
                print('--------------------------------------------')
                a = unprimed_letter
                b = unprimed_nato
                pa = a / (a + b) * 100
                pb = b / (a + b) * 100
                print(f'  letter| {a:3d}({pa:2.0f}%)\t{b:3d}({pb:2.0f}%)\t|{a + b}')  # noqa
                a = primed_letter
                b = primed_nato
                try:
                    pa = a / (a + b) * 100
                except ZeroDivisionError:
                    pa = 0.0
                try:
                    pb = b / (a + b) * 100
                except ZeroDivisionError:
                    pb = 0.0
                print(f'    nato| {a:3d}({pa:2.0f}%)\t{b:3d}({pb:2.0f}%)\t|{a + b}')  # noqa
            a = unprimed_letter + primed_letter
            b = unprimed_nato + primed_nato
            pa = a / (a + b) * 100
            pb = b / (a + b) * 100
            print('--------------------------------------------')
            print(f'   total| {a:3d}({pa:2.0f}%)\t{b:3d}({pb:2.0f}%)\t|{a + b}')  # noqa

    def plot_postcode_priming(self):
        """ Show the plot for spelling priming

        Colours = how the users where primed
        Intensity = higher when user behaviour aligns more with priming
        """

        def _get_node_id(_hear: str, _say: str, _layer: int):
            if _layer == 0:
                return 0
            elif 1 <= _layer <= 2:
                if _say == 'nato':
                    _id = 1
                elif _say == 'letter':
                    _id = 2
                elif _say == 'other':
                    _id = 3
                else:
                    raise ValueError(f'Error at layer {_layer}')
                return _id + (_layer - 1) * 3
            elif _layer == 3:
                if _hear == 'nato':
                    _id = 7
                elif _hear == 'letter':
                    _id = 8
                else:
                    raise ValueError(f'Error at layer {_layer}')
                return _id
            elif _layer >= 4:
                if (_hear, _say) == ('nato', 'nato'):
                    _id = 9
                elif (_hear, _say) == ('letter', 'nato'):
                    _id = 10
                elif (_hear, _say) == ('nato', 'letter'):
                    _id = 11
                elif (_hear, _say) == ('letter', 'letter'):
                    _id = 12
                elif _say == 'other':
                    _id = 13
                else:
                    raise ValueError(f'Error at layer {_layer}')
                return _id + (_layer - 4) * 5
            else:
                raise ValueError(f'Error at layer {_layer}')

        links = []
        # LAYER 0 -> 1 (postcode 1st attempt)
        for to in ['nato', 'letter', 'other']:
            # outbounds
            links.append(
                _SankeyLink(
                    from_=0, to=_get_node_id('none', to, 1),
                    value=sum(
                        self.spelling_transitions[(0, to, to_next, 'letter')]
                        for to_next in ['nato', 'letter', 'other']
                    ),
                    colour=_grey(0.5),
                ),
            )
        # LAYER 1 -> 2 (postcode 2nd attempt)
        for layer in [1]:
            for from_ in ['nato', 'letter', 'other']:
                for to in ['nato', 'letter', 'other']:
                    links.append(
                        _SankeyLink(
                            from_=_get_node_id('none', from_, layer),
                            to=_get_node_id('none', to, layer + 1),
                            value=self.spelling_transitions[
                                (0, from_, to, 'letter')
                            ],
                            colour=_grey(0.5),
                        )
                    )
        # LAYER 2 -> 3 priming
        for layer in [2]:
            for from_ in ['nato', 'letter', 'other']:
                for to in ['nato', 'letter']:
                    links.append(
                        _SankeyLink(
                            from_=_get_node_id('none', from_, layer),
                            to=_get_node_id(to, 'none', layer + 1),
                            value=(
                                self.spelling_transitions[
                                    (layer - 1, from_, 'letter', to)
                                ] + self.spelling_transitions[
                                    (layer - 1, from_, 'nato', to)
                                ] + self.spelling_transitions[
                                    (layer - 1, from_, 'other', to)
                                ]
                            ),
                            colour=_grey(0.5),
                            label=(
                                'Bravo👂🏽'
                                if to == 'nato'
                                else 'B👂🏽'
                            )
                        )
                    )
        # LAYER 3 -> 4 (postcode 3rd attempt)
        for layer in [3]:
            for from_ in ['nato', 'letter']:
                for to in ['nato', 'letter', 'other']:
                    links.append(
                        _SankeyLink(
                            from_=_get_node_id(from_, from_, layer),
                            to=_get_node_id(from_, to, layer + 1),
                            value=(
                                self.spelling_transitions[
                                    (layer - 1 - 1, "nato", to, from_)
                                ]
                                + self.spelling_transitions[
                                    (layer - 1 - 1, "letter", to, from_)
                                ]
                                + self.spelling_transitions[
                                    (layer - 1 - 1, "other", to, from_)
                                ]
                            ),
                            colour=_get_colour(
                                from_, to, from_,
                            ),
                            label=''
                        )
                    )

        # LAYER 4 -> 5 (name spelling)
        for layer in [4]:
            for prime in ['letter', 'nato']:
                for from_ in ['nato', 'letter', 'other']:
                    for to in ['nato', 'letter', 'other']:
                        links.append(
                            _SankeyLink(
                                from_=_get_node_id(prime, from_, layer),
                                to=_get_node_id(prime, to, layer + 1),
                                value=self.spelling_transitions[
                                    (layer - 1 - 1, from_, to, prime)
                                ],
                                colour=_get_colour(
                                    from_, to, prime,
                                ),
                                label=(
                                    'heard: "Bravo"'
                                    if prime == 'nato'
                                    else 'heard: "B"'
                                )
                            )
                        )

        for _i in range(19, 23, 1):
            links.append(
                _SankeyLink(
                    from_=_i,
                    to=_i + 1,
                    value=120.0,
                    colour=_grey(0.0),
                    label=""
                )
            )

        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement='perpendicular',
                    node=dict(
                        pad=15,
                        thickness=35,
                        line=dict(color="white", width=0.0),
                        label=(
                            ['']
                            + ['️Bravo', '️B', '?'] * 2
                            + ['Bravo', 'B']
                            + [
                                '️Bravo',  # (↑↑)
                                '️Bravo',  # (↑↓)
                                '️B',  # (↑↓)
                                '️B',  # (↑↑)
                                '?'
                            ] * 2
                            + ['Q1', 'Q2', 'prime', 'Q3', 'Q6']
                        ),
                        color=(
                            [
                                'white'
                            ] + [
                                'rgba(140, 140, 140, 1)',
                                _grey(0.8), _grey(0.8)
                            ] * 2
                            + [
                                _grey(0.8), _grey(0.8)
                            ] + [
                                _red(0.8), _blue(0.8),
                                _blue(0.8), _red(0.8),
                                _grey(0.5)
                            ] * 2
                            + [_grey(0.0)] * 5
                        )
                    ),
                    textfont=dict(size=28),
                    orientation="v",
                    link=dict(
                        source=[_l.from_ for _l in links][3:],
                        target=[_l.to for _l in links][3:],
                        value=[_l.value for _l in links][3:],
                        color=[_l.colour for _l in links][3:],
                        label=[_l.label for _l in links][3:],
                    )
                )
            ])
        fig.update_layout(
            title_text=f"Speaking behaviour (spelling) — {self.locale}",
            font_size=34, title_x=0.5
        )
        fig.show()

    def plot_month_priming(self):
        """ Show the plot for month priming

        Colours = how the users where primed
        Intensity = higher when user behaviour aligns more with priming
        """

        def _get_node_id(_read: str, _say: str, _layer: int):
            if _layer == 0:
                if _read == 'word':
                    return 0
                elif _read == 'digit':
                    return 1
                else:
                    raise ValueError('Error at layer 0 (priming)')
            elif _layer >= 1:
                if (_read, _say) == ('word', 'word'):
                    _id = 2
                elif (_read, _say) == ('digit', 'word'):
                    _id = 3
                elif (_read, _say) == ('word', 'digit'):
                    _id = 4
                elif (_read, _say) == ('digit', 'digit'):
                    _id = 5
                elif _say == 'other':
                    _id = 6
                else:
                    raise ValueError(f'Error at layer {_layer}')
                return _id + (_layer - 1) * 5
            else:
                raise ValueError()

        links = []
        word_january = _get_word_january(self.locale)

        # LAYER 0 -> 1 (date 1st attempt)
        # primed with [read Jan] -> first attempt
        for layer in [0]:
            for prime in ['word', 'digit']:
                for from_ in [prime]:
                    for to in ['word', 'digit', 'other']:
                        links.append(
                            _SankeyLink(
                                from_=_get_node_id(prime, from_, layer),
                                to=_get_node_id(prime, to, layer + 1),
                                value=self.month_transitions[
                                    (5 + layer, from_, to, prime)
                                ],
                                colour=_get_colour('start', to, prime),
                                label=(
                                    f'{word_january}'
                                    if prime == 'word'
                                    else '1'
                                )
                            )
                        )

        # # LAYER 1 -> 2 (date 2nd attempt) & LAYER 2 -> 3 (date 3rd attempt)
        for layer in [1, 2]:
            for prime in ['word', 'digit']:
                for from_ in ['word', 'digit', 'other']:
                    for to in ['word', 'digit', 'other']:
                        links.append(
                            _SankeyLink(
                                from_=_get_node_id(prime, from_, layer),
                                to=_get_node_id(prime, to, layer + 1),
                                value=self.month_transitions[
                                    (5 + layer, from_, to, prime)
                                ],
                                colour=_get_colour(from_, to, prime),
                                label=(
                                    f'{word_january}'
                                    if prime == 'word'
                                    else '1'
                                )
                            )
                        )

        for _i in range(17, 20, 1):
            links.append(
                _SankeyLink(
                    from_=_i,
                    to=_i + 1,
                    value=150.0,
                    colour=_grey(0.0),
                    label=""
                )
            )

        word_one = _get_word_one(self.locale)
        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement='perpendicular',
                    node=dict(
                        pad=15,
                        thickness=35,
                        line=dict(color="white", width=0.0),
                        label=(
                            [
                                f'{word_january}', '1',
                            ] + [
                                f'️{word_january}',  # (↑↑)
                                f'{word_january}',  # (↑↓)
                                f'️{word_one}',  # (↑↓)
                                f'️{word_one}',  # (↑↑)
                                '?'
                            ] * 3 + [
                                'prime', 'Q7', 'Q8', 'Q9'
                            ]
                        ),
                        color=(
                            [
                                _grey(0.8), _grey(0.8),
                            ] + [
                                _red(0.8), _blue(0.8),
                                _blue(0.8), _red(0.8),
                                _grey(0.5)
                            ] * 3
                            + [_grey(0.0)] * 4
                        )
                    ),
                    orientation="v",
                    textfont=dict(size=26),
                    link=dict(
                        source=[_l.from_ for _l in links],
                        target=[_l.to for _l in links],
                        value=[_l.value for _l in links],
                        color=[_l.colour for _l in links],
                        label=[_l.label for _l in links],
                    )
                )
            ])
        fig.update_layout(
            title_text=f"Speaking behaviour (date) — {self.locale}",
            font_size=30, title_x=0.5,
        )
        fig.show()

    def print_name_priming(self):
        """ Print priming analysis concerning names"""
        print('********************************************')
        print('*************** Name priming ***************')
        print('********************************************')
        for i in range(3):
            print('--------------------------------------------')
            print(f'-                Attempt {i + 1}                 -')
            print('--------------------------------------------')
            primed_nato = self.prime_name[(i, 'nato', True)]
            primed_letter = self.prime_name[(i, 'nato', False)]
            unprimed_nato = self.prime_name[(i, 'letter', True)]
            unprimed_letter = self.prime_name[(i, 'letter', False)]
            print('Hear\\Say| letter\tnato\t\t|total')
            print('--------------------------------------------')
            a = unprimed_letter
            b = unprimed_nato
            pa = a / (a + b) * 100
            pb = b / (a + b) * 100
            print(f'  letter| {a:3d}({pa:2.0f}%)\t{b:3d}({pb:2.0f}%)\t|{a + b}')  # noqa
            a = primed_letter
            b = primed_nato
            try:
                pa = a / (a + b) * 100
            except ZeroDivisionError:
                pa = 0.0
            try:
                pb = b / (a + b) * 100
            except ZeroDivisionError:
                pb = 0.0
            print(f'    nato| {a:3d}({pa:2.0f}%)\t{b:3d}({pb:2.0f}%)\t|{a + b}')  # noqa
            a = unprimed_letter + primed_letter
            b = unprimed_nato + primed_nato
            pa = a / (a + b) * 100
            pb = b / (a + b) * 100
            print('--------------------------------------------')
            print(f'   total| {a:3d}({pa:2.0f}%)\t{b:3d}({pb:2.0f}%)\t|{a + b}')  # noqa

    def print_dob_priming(self):
        """ Print priming analysis concerning date of birth"""
        print('********************************************')
        print('*************** DOB priming ****************')
        print('********************************************')
        for i in range(3):
            print('--------------------------------------------')
            print(f'-                Attempt {i + 1}                 -')
            print('--------------------------------------------')
            primed_month = self.prime_dob[(i, 'word', True)]
            primed_number = self.prime_dob[(i, 'word', False)]
            unprimed_month = self.prime_dob[(i, 'digit', True)]
            unprimed_number = self.prime_dob[(i, 'digit', False)]
            print('See\\Say | number\tmonth\t\t|total')
            print('--------------------------------------------')
            a = unprimed_number
            b = unprimed_month
            pa = a / (a + b) * 100
            pb = b / (a + b) * 100
            print(f'  number| {a:3d}({pa:2.0f}%)\t{b:3d}({pb:2.0f}%)\t|{a + b}')  # noqa
            a = primed_number
            b = primed_month
            pa = a / (a + b) * 100
            pb = b / (a + b) * 100
            print(f'   month| {a:3d}({pa:2.0f}%)\t{b:3d}({pb:2.0f}%)\t|{a + b}')  # noqa
            a = unprimed_number + primed_number
            b = unprimed_month + primed_month
            pa = a / (a + b) * 100
            pb = b / (a + b) * 100
            print('--------------------------------------------')
            print(f'   total| {a:3d}({pa:2.0f}%)\t{b:3d}({pb:2.0f}%)\t|{a + b}')  # noqa

    def print_priming_analysis(self):
        """ Print summary of dialogue data analysis wrt priming """
        print('============================================')
        print('             Priming Analysis               ')
        print('============================================')
        if self.email_mode:
            self.print_email_priming()
        else:
            self.print_postcode_priming()
            self.print_name_priming()
            self.print_dob_priming()


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--locale', type=str, default='en_GB',
        choices={"en_GB", "pl_PL", "fr_FR", "el_GR"},
        help='Locale to evaluate'
    )
    parser.add_argument(
        '--email', action="store_true",
        help='Whether to switch to mode compatible with email analysis'
    )
    args = parser.parse_args()
    #
    locale = args.locale.replace("_", "-")
    lang_code = locale.split('-')[0]
    #
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
    dialogue_id2turns = read_dialogues(dialogues_file)
    #
    profile_analyser = ProfileAnalyser()
    name_parser_analyser = NameParserAnalyser(
        parser=EviNameParser(
            locale=locale,
            strict=True,
            use_nbest=True,
        )
    )
    for _, p in scenario_id2profile.items():
        profile_analyser.increment(p)
        name_parser_analyser.increment(p)
    #
    dialogue_analyser = DialogueAnalyser(
        locale=locale,
        scenario_id2profile=scenario_id2profile,
        email_mode=args.email
    )
    for _, turns in dialogue_id2turns.items():
        dialogue_analyser.increment(turns)
    #
    dialogue_analyser.print_asr_analysis()
    dialogue_analyser.print_priming_analysis()
    profile_analyser.print_profile_analysis()
    name_parser_analyser.print_parsing_analysis()
    dialogue_analyser.print_dialogue_analysis()
    if lang_code == 'en':
        dialogue_analyser.plot_postcode_priming()
    dialogue_analyser.plot_month_priming()


if __name__ == '__main__':
    _main()
    print("Done!")
