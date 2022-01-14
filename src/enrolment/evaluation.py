""" Evaluator for enrolment

Copyright PolyAI Limited
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from data_types import Profile
from utils.math_ops import calc_f1, safe_div


class EnrolmentEvaluator(object):
    """ Evaluator for Enrolment task """

    def __init__(self, slots: List[str]):
        """ Initialise

        Args:
            slots: slots to enroll
        """
        self.pred_profiles = []
        self.gold_profiles = []
        self.slot2all_n_attempts = defaultdict(list)
        self.slots = slots

    def increment(
        self,
        pred_profile: Profile,
        gold_profile: Profile,
        slot2n_attempts: Dict[str, int]
    ):
        """ Add a predictions / ground truth pair

        Args:
            pred_profile: the extracted profile
            gold_profile: the ground truth profile
            slot2n_attempts: dict slot->num of collection attempts
        """
        self.pred_profiles.append(pred_profile)
        self.gold_profiles.append(gold_profile)
        for slot, n in slot2n_attempts.items():
            self.slot2all_n_attempts[slot].append(n)

    def _compare_wrt_item(
        self, item: str, profile: Profile, other: Profile
    ) -> bool:
        return profile.get_item_hash(item) == other.get_item_hash(item)

    def _compare_profiles(self, profile: Profile, other: Profile) -> bool:
        res = True
        for slot in self.slots:
            if not profile.has(slot):
                raise ValueError(f"Missing value for slot {slot}")
            val = self._compare_wrt_item(
                item=slot, profile=profile, other=other
            )
            res = res and val
        return res

    def _calc_slot_metrics(self, slot: str) -> Tuple[float, float, float]:
        n_correct = 0
        n_predictions = 0
        n_targets = 0
        for pred, gold in zip(self.pred_profiles, self.gold_profiles):
            n_targets += 1
            if pred.has(item=slot):
                n_correct += self._compare_wrt_item(slot, pred, gold)
                n_predictions += 1
        p = safe_div(n_correct, n_predictions)
        r = safe_div(n_correct, n_targets)
        f1 = calc_f1(p, r)
        return p, r, f1

    def _calc_profile_metrics(self) -> Tuple[float, float, float]:
        n_correct = 0
        n_predictions = 0
        n_targets = 0
        for pred, gold in zip(self.pred_profiles, self.gold_profiles):
            n_targets += 1
            try:
                n_correct += self._compare_profiles(pred, gold)
                n_predictions += 1
            except ValueError:
                continue
        p = safe_div(n_correct, n_predictions)
        r = safe_div(n_correct, n_targets)
        f1 = calc_f1(p, r)
        return p, r, f1

    def _get_attempts(self, slot: Optional[str]) -> np.array:
        if slot is None:
            # sum for all slots
            n_part_attempts = []
            for _, n_slot_attempts in self.slot2all_n_attempts.items():
                n_part_attempts.append(n_slot_attempts)
            n_part_attempts = np.vstack(n_part_attempts)
            n_all_attempts = np.sum(n_part_attempts, axis=0)
        else:
            n_all_attempts = np.asarray(self.slot2all_n_attempts[slot])
        return n_all_attempts

    def _calc_attempts(
        self,
        slot: Optional[str]
    ) -> Tuple[float, float, float, float]:
        n_all_attempts = self._get_attempts(slot)
        min_ = np.min(n_all_attempts).item()
        med = np.median(n_all_attempts).item()
        avg = np.mean(n_all_attempts).item()
        max_ = np.max(n_all_attempts).item()
        return min_, med, avg, max_

    def _calc_flow(
        self,
        slot: Optional[str],
        normalise: bool = False
    ) -> List[int]:
        """Calculate how many callers would reach each turn"""
        n_reached_turn = []
        n_all_attempts = self._get_attempts(slot)
        _max = np.max(n_all_attempts).item()
        for n_turn in range(1, _max + 1):
            n_reached_turn.append(
                np.sum(n_all_attempts >= n_turn).item()
            )
        if normalise:
            denom = max(n_reached_turn + [1])
            n_reached_turn = [n / denom for n in n_reached_turn]
        return n_reached_turn

    def print_report(self, draw_plots: bool = True):
        """ Print evaluation report for enrolment task """
        print('=====================')
        print('Enrolment Report')
        print('=====================')
        print(f'n_data: \t{len(self.pred_profiles)}')
        for slot in self.slots:
            slot_p, slot_r, slot_f1 = self._calc_slot_metrics(slot)
            n_min, n_med, n_avg, n_max = self._calc_attempts(slot)
            n_at_turn = self._calc_flow(slot, normalise=False)
            p_at_turn = self._calc_flow(slot, normalise=True)
            print(f'---{slot.upper()}---')
            print(f'%at turn:  {">".join(f"{p * 100:.0f}" for p in p_at_turn)}')  # noqa
            print(f'#at turn:  {">".join(str(n) for n in n_at_turn)}')
            print(f'#attempts: {n_min}(min) {n_med}(med) '
                  f'{n_avg:.2f}(avg) {n_max}(max)')
            print(f'P\t{slot_p * 100:.2f}%')
            print(f'R\t{slot_r * 100:.2f}%')
            print(f'F1\t{slot_f1 * 100:.2f}%')
        #
        profile_p, profile_r, profile_f1 = self._calc_profile_metrics()
        n_min, n_med, n_avg, n_max = self._calc_attempts(None)
        print('----PROFILE---')
        print(f'attempts: {n_min}(min) {n_med}(med) '
              f'{n_avg:.2f}(avg) {n_max}(max)')
        print(f'P\t{profile_p * 100:.2f}%')
        print(f'R\t{profile_r * 100:.2f}%')
        print(f'F1\t{profile_f1 * 100:.2f}%')
        print('---------------------')
        if draw_plots:
            self.plot_collection_flow()

    def plot_collection_flow(self):
        """ Plot flow of collection for each item """
        # local imports so we don't need to declare in BUILD

        import plotly.graph_objects as go

        @dataclass()
        class _SankeyLink:
            from_: int
            to: int
            value: float
            colour: str
            label: str = ''

        slot2rgb = {
            'postcode': '255, 20, 147',
            'name': '20, 147, 255',
            'dob': '147, 255, 20',
        }

        links = []
        labels = []
        for slot in self.slots:
            p_at_turn = self._calc_flow(slot, normalise=False)
            labels.append('')
            for i, p in enumerate(p_at_turn):
                c = f'rgba({slot2rgb.get(slot, "255,255,255")}, 0.4)'
                if i == 0:
                    links.append(
                        _SankeyLink(
                            from_=len(labels) - 1,
                            to=len(labels),
                            value=p,
                            colour=c,
                        )
                    )
                else:
                    links.append(
                        _SankeyLink(
                            from_=len(labels) - 1,
                            to=len(labels),
                            value=p,
                            label=f'retry #{i}',
                            colour=c
                        )
                    )
                labels.append(slot)

        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement='perpendicular',
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="white", width=0.0),
                        label=labels,
                        color=[
                            f'rgba({slot2rgb.get(_l, "255,255,255")}, 0.8)'
                            for _l in labels
                        ]
                    ),
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
            title_text="Collection Flow",
            font_size=14,
        )
        fig.show()
