""" Evaluators for tasks

Copyright PolyAI Limited
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import glog
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, det_curve

from readers import Profile, Turn


def _calc_f1(p: float, r: float) -> float:
    nom = 2 * (p * r)
    denom = (p + r)
    if not denom:
        return 0
    return nom / denom


def _safe_div(nom: float, denom: float, default: float = 0.0) -> float:
    try:
        return nom / denom
    except ZeroDivisionError as e:
        glog.warn(str(e))
        return default


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
        p = _safe_div(n_correct, n_predictions)
        r = _safe_div(n_correct, n_targets)
        f1 = _calc_f1(p, r)
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
        p = _safe_div(n_correct, n_predictions)
        r = _safe_div(n_correct, n_targets)
        f1 = _calc_f1(p, r)
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


class VerificationEvaluator(object):
    """ Evaluator for Verification task """

    def __init__(self):
        """ Initialise """
        self.pred_scores = []
        self.golds = []
        self.n_nones = 0
        self.n_turns = []

    def increment(
        self,
        pred_score: Optional[float],
        is_genuine: bool,
        n_turns: int
    ):
        """ Add a predictions / ground truth pair

        Args:
            pred_score: the predicted verification score
            is_genuine: whether this is a genuine attempt (or impostor)
            n_turns: number of turns that verification took
        """
        self.n_turns.append(n_turns)
        if pred_score is None:
            self.n_nones += 1
        else:
            self.pred_scores.append(pred_score)
            self.golds.append(is_genuine)

    def _calc_avg_turns(self) -> float:
        return np.mean(self.n_turns).item()

    def calc_fta(self) -> float:
        """ Failure To Acquire """
        return self.n_nones / (len(self.pred_scores) + self.n_nones)

    def calc_det(self) -> Tuple[List[float], List[float], List[float]]:
        """ Detection Error Tradeoff """
        # Sklearns DET curve is the curve
        # x = FPR (False Positive Rate)
        # y = FNR (False negative Rate)
        # Evaluated at points that correspond to different thresholds
        fprs, fnrs, thresholds = det_curve(
            y_true=self.golds,
            y_score=self.pred_scores
        )
        # Biometric evaluations use different terminology:
        # FMR (False Match Rate) = FPR = Proportion of
        # impostor attempts that are falsely declared
        # to match a template of another object
        # and
        # FNMR (False Non-Match Rate) = FNRS = Proportion of
        # genuine attempts that are falsely declared
        # not to match a template of the same object
        fmrs = fprs.tolist()
        fnmrs = fnrs.tolist()
        thresholds = thresholds.tolist()
        # FTA (Failure-to-Acquire Rate) = Proportion of
        # the attempts for which the system fails
        # to producea sample of sufficient quality
        fta = self.calc_fta()
        # Finally, the DET curve for (biometric) evaluation is the curve
        # x = FAR (False Accept Rate)
        # y = FRR (False Reject Rate)
        # These are roughly the same as FMR and FNMR respectively,
        # but the definition distinguishes between attempts and transactions.
        # A transaction may consist of a sequence of attempts and
        # depending on the system's configuration the outcome of
        # individual attempts affects the transaction differently.
        fars = [
            self._calc_far(fta=fta, fmr=fmr)
            for fmr in fmrs
        ]
        frrs = [
            self._calc_frr(fta=fta, fnmr=fnmr)
            for fnmr in fnmrs
        ]
        return fars, frrs, thresholds

    def _plot_det(
        self,
        fars: List[float],
        frrs: List[float],
        thresholds: List[float],
        logscale: bool = True,
    ):
        assert (len(fars) == len(frrs) == len(thresholds))
        xs, ys = zip(*sorted(
            list(zip(fars, frrs)),
            key=lambda x: (x[0], -x[1]))
        )
        plt.figure()
        plt.plot(
            [x * 100 for x in xs] if not logscale else xs,
            [y * 100 for y in ys],
            'go--',
            linewidth=2,
        )
        if logscale:
            plt.xscale('log')
            plt.xlabel("False Acceptance Rate")
        else:
            # plt.plot([0, 100], [0, 100], 'b:')
            plt.xlabel("False Acceptance Rate (%)")
            pass
        plt.ylabel("False Rejection Rate (%)")
        plt.title("Detection Error Trade-off")
        plt.show()

    def _calc_eer(
        self,
        fars: List[float],
        frrs: List[float],
        thresholds: List[float],
    ) -> Tuple[float, float]:
        """ Equal Error Rate

        Reference:
        Maio, D., Maltoni, D., Cappelli, R., Wayman, J. L., & Jain, A. K.
        (2002). FVC2000: Fingerprint verification competition.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        24(3), 402-412.
        """
        fars = np.asarray(fars)
        frrs = np.asarray(frrs)
        thresholds = np.asarray(thresholds)
        #
        t1 = np.nanargmax(thresholds * np.where(frrs <= fars, 1, np.nan))
        t2 = np.nanargmin(thresholds * np.where(frrs >= fars, 1, np.nan))
        if frrs[t1] + fars[t1] <= frrs[t2] + fars[t2]:
            eer_low, eer_high = frrs[t1], fars[t1]
        else:
            eer_low, eer_high = fars[t2], frrs[t2]
        eer_threshold = np.mean([thresholds[t1], thresholds[t2]])
        eer_value = np.mean([eer_low, eer_high])
        return float(eer_value), float(eer_threshold)

    def _calc_frr_at_far(
        self,
        fars: List[float],
        frrs: List[float],
        thresholds: List[float],
        far_operational: float
    ) -> Tuple[float, float]:
        """Find a point with a given FAR on the DET curve"""
        # reverse DET curve, so it is in ascending order of FAR
        fars = np.asarray(fars)[::-1]
        frrs = np.asarray(frrs)[::-1]
        thresholds = np.asarray(thresholds)[::-1]
        # find FRR for given FAR (estimate via linear interpolation)
        frr_operational = np.interp(
            far_operational,
            fars, frrs,
            left=np.max(frrs), right=0.0
        )
        # find threshold for given FAR (estimate via linear interpolation)
        threshold_operational = np.interp(
            far_operational,
            fars, thresholds,
            left=1.0, right=0.0
        )
        return float(frr_operational), float(threshold_operational)

    def _calc_frr(self, fta: float, fnmr: float) -> float:
        """ False Rejection Rate """
        frr = fta + fnmr * (1 - fta)
        return frr

    def _calc_far(self, fta: float, fmr: float) -> float:
        """ False Acceptance Rate """
        far = fmr * (1 - fta)
        return far

    def print_report(self, draw_plots: bool = True):
        """ Print evaluation report for verification task """
        avg_turns = self._calc_avg_turns()
        fta = self.calc_fta()
        fars, frrs, thresholds = self.calc_det()
        eer, eer_thesh = self._calc_eer(
            fars=fars, frrs=frrs,
            thresholds=thresholds
        )
        attempts2operational = {}
        for n_attempts in [10, 100, 1000, 10000, float("inf")]:
            frr_operational, threshold_operational = self._calc_frr_at_far(
                fars=fars, frrs=frrs,
                thresholds=thresholds,
                far_operational=1 / n_attempts
            )
            attempts2operational[n_attempts] = (
                frr_operational, threshold_operational
            )
        n_genuine = sum(g for g in self.golds)
        n_impostor = sum(not g for g in self.golds)
        print('=====================')
        print('Verification Report')
        print('=====================')
        print(f'#attempts: \t{n_genuine + n_impostor}')
        print(f'#genuine: \t{n_genuine}')
        print(f'#impostor: \t{n_impostor}')
        print(f'avg turns:\t {avg_turns:.2f}')
        print(f'FTA: \t{fta * 100:.2f}%')
        print(f'EER: \t{eer * 100:.2f}% (@θ={eer_thesh:.2f})')
        for n_attempts, (frr_operational, threshold_operational) in sorted(
            attempts2operational.items(), key=lambda x: x[0]
        ):
            print(f'FRR {frr_operational * 100:6.2f}% @ FAR 1 / {n_attempts} (@θ={threshold_operational:.4f})')  # noqa
        print('---------------------')
        if draw_plots:
            self._plot_det(fars=fars, frrs=frrs, thresholds=thresholds)
            filepath = './det.tsv'
            with open(filepath, 'w') as fout:
                for x, y in zip(fars, frrs):
                    fout.write(f'{x}\t{y}\n')


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
            p_no_pred = _safe_div(n_errors_no_pred, n_errors)
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
