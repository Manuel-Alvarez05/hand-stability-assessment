# core/plotting_utils.py

"""
This module contains helper functions for creating plots of:
- Fingertip displacement over time
- Fatigue indices (bar chart)
- Optional correlation / similarity between fingers

All plotting should be:
- Simple
- Readable in a clinical / dashboard context
- Consistent with the color palette defined in core.config

We will generally return matplotlib Figure objects so Streamlit can display
them with st.pyplot(fig).
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from core import config


def plot_displacement_time_series(displacement_ts: Dict[str, List[Tuple[float, float]]]):
    """
    Create a line plot of displacement vs. time for each finger.

    Parameters
    ----------
    displacement_ts : dict
        Dictionary mapping finger name -> list of (t, displacement) samples.

        Example:
        {
            "THUMB": [(t0, d0), (t1, d1), ...],
            "INDEX": [(t0, d0), (t1, d1), ...],
            "MIDDLE": [(t0, d0), (t1, d1), ...]
        }

    TODO (AI / TEAM):
    - Use matplotlib to create a figure.
    - For each finger in displacement_ts:
        * Extract time (t) and displacement (d) values.
        * Plot t vs. d as a line.
        * Use config.FINGER_COLORS[finger] for the line color.
    - Label axes appropriately.
    - Add a legend and title.
    - Return the matplotlib Figure object.
    """
    raise NotImplementedError
