""" Evaluators for tasks

Copyright PolyAI Limited
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score

from readers import Profile, Turn
from utils.math_ops import safe_div


class IdentificationEvaluator(object):
    """ Evaluator for Identification task """

    def __init__(
        self,
        scenario_id2profile: Dict[str, Profile],
        dialogue_id2turns: Dict[str, List[Turn]]
    ):
        """ Initialise

        Args:
            scenario_id2profile: a dict from scenario id to profile
            dialogue_id2turns: a dict from dialogue id to list of turns
        """
        self.preds = []
        self.golds = []
        self.n_turns = []
        self.dialogues = []
        self.scenario_id2profile = scenario_id2profile
        self.dialogue_id2turns = dialogue_id2turns

    def increment(
        self,
        gold_id: str,
        pred_id: str,
        dialogue_id: str,
        n_turns: int
    ):
        """ Add a predictions / ground truth pair

        Args:
            gold_id: the ground truth profile id
            pred_id: the identified profile id
            dialogue_id: the id of the dialogue this sample came from
            n_turns: number of turns that identification took
        """
        self.golds.append(gold_id)
        self.preds.append(pred_id)
        self.n_turns.append(n_turns)
        self.dialogues.append(dialogue_id)

    def calc_identification_rate(self) -> float:
        """ Identification rate (at rank 1) """
        a = accuracy_score(y_true=self.golds, y_pred=self.preds)
        return a

    def _calc_avg_turns(self) -> float:
        return np.mean(self.n_turns).item()

    def _print_errors(self):
        print('=====================')
        print('Identification Errors')
        print('=====================')
        n_errors = 0
        n_errors_no_pred = 0
        for dialogue_id, pred_id, gold_id in zip(
            self.dialogues, self.preds, self.golds
        ):
            if pred_id == gold_id:
                # not an error
                continue
            #
            n_errors += 1
            try:
                pred = self.scenario_id2profile[pred_id]
            except KeyError:
                # TYPE OF ERROR: couldn't identify any profile
                n_errors_no_pred += 1
                pred = Profile(
                    scenario_id=pred_id,
                    postcode='NONE',
                    name_first='NONE',
                    name_last='NONE',
                    dob_str='NONE'
                )
            gold = self.scenario_id2profile[gold_id]
            attribute_gold_vs_pred = []
            for attribute in [
                'postcode', 'name_first', 'name_last', 'dob_str'
            ]:
                p = pred.__dict__[attribute]
                g = gold.__dict__[attribute]
                if p != g:
                    attribute_gold_vs_pred.append((attribute, g, p))
            print(
                f'Misidentified {gold_id} as {pred_id} '
                f'(dialogue: {dialogue_id})'
            )
            for i, t in enumerate(self.dialogue_id2turns[dialogue_id]):
                print(f'{i}\t{t.transcription}')
            for a, g, p in attribute_gold_vs_pred:
                print(f'{a:15s}\tgold:{g}\tpred:{p}')
            print('*****')
            #
            p_no_pred = safe_div(n_errors_no_pred, n_errors)
            p_other = 1.0 - p_no_pred
            print("-----Error breakdown-----")
            print(f"#misidentifications: {n_errors}")
            print(f"Identified NONE (%): {p_no_pred * 100:.2f}")
            print(f"Identified WRONG(%): {p_other * 100:.2f}")

    def print_report(self):
        """ Print evaluation report for identification task """
        ir = self.calc_identification_rate()
        avg_turns = self._calc_avg_turns()
        self._print_errors()
        print('=====================')
        print('Identification Report')
        print('=====================')
        print(f'n_data: \t{len(self.preds)}')
        print(f'IR@1:   \t{ir * 100:.2f}%')
        print(f'avg turns:\t {avg_turns:.2f}')
        print('---------------------')
