import streamlit as st
from core import config
# from core import mediapipe_utils, signal_processing   # TO BE IMPLEMENTED

st.set_page_config(page_title="Live Test", page_icon="📊", layout="wide")

st.title("Step 2: Live Stability Test")

if "baseline_positions" not in st.session_state:
    st.error("Please complete Calibration first.")
    st.stop()

st.markdown(
    f"""
    Hold your hand in the **same position** as during calibration.

    We will record **{config.TEST_DURATION_SECONDS} seconds** of fingertip motion
    to estimate tremor, drift, and fatigue.
    """
)

st.divider()

if "raw_time_series" not in st.session_state:
    # Example structure to be filled by AI:
    st.session_state["raw_time_series"] = {
        # "THUMB": list of (t, x, y)
        # "INDEX": ...
        # "MIDDLE": ...
    }

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Webcam & Landmark Tracking")

    # TODO (AI / TEAM):
    # - Start a timed capture loop (~TEST_DURATION_SECONDS).
    # - At each time step:
    #   * Capture frame from webcam
    #   * Detect hand landmarks via MediaPipe
    #   * Extract fingertip coordinates for THUMB, INDEX, MIDDLE
    #   * Append (time, x, y) to st.session_state["raw_time_series"][finger]
    # - Optionally, draw small circles or markers on each fingertip in the video.
    # - Show the video frames in Streamlit during capture.
    st.info("Live webcam with fingertip tracking will be rendered here.")

with col2:
    st.subheader("Test Control & Status")

    if st.button("▶ Start 30s Test"):
        # TODO (AI / TEAM):
        # - Implement the timed capture here.
        # - Use Python's time module or a loop to approximate real time.
        # - Populate st.session_state["raw_time_series"] with all captured data.
        # - After capture, set a flag like st.session_state["test_complete"] = True.
        st.session_state["test_complete"] = True
        st.success("Test complete. Raw data has been captured. (Logic to be added.)")

    if st.session_state.get("test_complete"):
        st.success("Test complete! Proceed to the Results page.")
        st.caption("Use the sidebar to go to '3_Results'.")
