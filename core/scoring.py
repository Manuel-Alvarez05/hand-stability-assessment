# core/scoring.py

from core import config

def compute_stability_score(tremor, drift, fatigue):
    """
    tremor, drift, fatigue: dicts of finger -> metric value

    TODO (AI / TEAM):
    - Aggregate each metric across fingers (e.g., average).
    - Normalize each metric into a 0–1 penalty score.
    - Combine using config.WEIGHT_TREMOR, WEIGHT_DRIFT, WEIGHT_FATIGUE.
    - Map final penalty into a 0–100 "Stability Score" where:
        100 = very stable, 0 = very unstable.
    - Return the scalar stability score, plus any intermediate values if useful.
    """
    raise NotImplementedError
