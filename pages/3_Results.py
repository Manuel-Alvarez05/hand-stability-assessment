import streamlit as st
from core import config
# from core import signal_processing, scoring, plotting_utils   # TO BE IMPLEMENTED

st.set_page_config(page_title="Results", page_icon="📈", layout="wide")

st.title("Step 3: Results & Interpretation")

if not st.session_state.get("test_complete"):
    st.error("You need to complete a Live Test before viewing results.")
    st.stop()

raw_data = st.session_state.get("raw_time_series", {})

# TODO (AI / TEAM):
# - Implement the following in signal_processing.py:
#   * compute_displacement_time_series(raw_data, baseline_positions)
#   * compute_tremor_metrics(...)
#   * compute_drift_metrics(...)
#   * compute_fatigue_metrics(...)
#
# Example pseudocode for what we want:
# displacement = signal_processing.compute_displacement_time_series(
#     raw_data, st.session_state["baseline_positions"]
# )
# tremor = signal_processing.compute_tremor_metrics(displacement)
# drift = signal_processing.compute_drift_metrics(displacement)
# fatigue = signal_processing.compute_fatigue_metrics(displacement)
#
# stability_score = scoring.compute_stability_score(tremor, drift, fatigue)

st.subheader("Summary Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Mean Tremor (all fingers)", "TODO", "units")
with col2:
    st.metric("Mean Drift", "TODO", "units")
with col3:
    st.metric("Mean Fatigue Index", "TODO")
with col4:
    st.metric("Stability Score", "TODO", "0–100")

st.divider()

st.subheader("Displacement Over Time")
st.markdown("Plots of fingertip displacement relative to baseline for each finger.")

# TODO (AI / TEAM):
# - Use plotting_utils to generate matplotlib or Streamlit-native line plots.
# - One plot per finger, or overlay with different colors.

st.info("Line plots of displacement will appear here.")

st.subheader("Fatigue & Coordination")

# TODO (AI / TEAM):
# - Plot fatigue indices by finger as a bar chart.
# - Optionally, compute Pearson correlation between finger signals and plot as heatmap.

st.info("Bar chart for fatigue and optional correlation heatmap will appear here.")

st.divider()

st.subheader("Clinical-Style Interpretation")
st.markdown(
    """
    - **Tremor amplitude**: Higher values may correspond to less steady hands.
    - **Drift**: Large drift suggests difficulty maintaining a fixed posture.
    - **Fatigue index**: Values > 1 indicate increasing tremor over the test.
    - **Stability score**: Combines these into a single 0–100 index.

    ⚠️ **Important:** This is an educational simulation, not a diagnostic tool.
    """
)
