""" Math operators

Copyright PolyAI Limited
"""

import glog


def calc_f1(p: float, r: float) -> float:
    nom = 2 * (p * r)
    denom = (p + r)
    return safe_div(nom, denom, 0)


def safe_div(nom: float, denom: float, default: float = 0.0) -> float:
    try:
        return nom / denom
    except ZeroDivisionError as e:
        glog.warn(str(e))
        return default
