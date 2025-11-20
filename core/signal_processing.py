# core/signal_processing.py

"""
This module will convert raw fingertip time series into:
- Displacement relative to baseline.
- Tremor metrics (e.g., RMS).
- Drift metrics.
- Fatigue indices.
"""

def compute_displacement_time_series(raw_data, baseline_positions):
    """
    raw_data: dict of finger -> list of (t, x, y)
    baseline_positions: dict of finger -> (x0, y0)

    TODO (AI / TEAM):
    - For each finger:
        * For each sample, subtract baseline (x0, y0) from (x, y).
        * Optionally, convert to 1D displacement (e.g., Euclidean distance).
    - Return a dict with cleaned displacement time series.
    """
    raise NotImplementedError

def compute_tremor_metrics(displacement_ts):
    """
    displacement_ts: dict of finger -> list of (t, displacement)

    TODO (AI / TEAM):
    - For each finger:
        * Compute an overall measure of tremor amplitude
          (e.g., RMS of displacement).
    - Return a dict of tremor metrics per finger.
    """
    raise NotImplementedError

def compute_drift_metrics(displacement_ts):
    """
    TODO (AI / TEAM):
    - For each finger:
        * Compute slow drift away from baseline
          (e.g., difference between start and end positions).
    - Return a dict of drift metrics per finger.
    """
    raise NotImplementedError

def compute_fatigue_metrics(displacement_ts):
    """
    TODO (AI / TEAM):
    - For each finger:
        * Split the time series into early and late segments.
        * Compute tremor amplitude in early vs late segments.
        * Fatigue index = late_amp / early_amp.
    - Return a dict of fatigue indices per finger.
    """
    raise NotImplementedError
