""" Evaluators for tasks

Copyright PolyAI Limited
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import det_curve


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
