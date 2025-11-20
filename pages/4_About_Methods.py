import streamlit as st

st.set_page_config(page_title="About & Methods", page_icon="ℹ️", layout="wide")

st.title("About This Tool & Methods")

st.subheader("Biomedical Motivation")
st.markdown(
    """
    This project explores how **hand tremor, drift, and fatigue** can be quantified
    using computer vision, with potential relevance for:

    - Fine motor control tasks (e.g., surgery, microsuturing, instrument handling)
    - Age-related changes in motor function
    - Neuromotor conditions that affect tremor and stability
    """
)

st.subheader("MediaPipe Hand Tracking")
st.markdown(
    """
    We use **MediaPipe Hands** to detect hand landmarks from a standard webcam.
    MediaPipe provides 21 3D landmarks per hand, including fingertips.

    In this app, we focus on the **thumb, index, and middle fingertip landmarks**
    and track their movement over time during a 30-second hold task.
    """
)

st.subheader("Signal Processing (Planned)")
st.markdown(
    """
    For each tracked fingertip, we compute:

    - **Displacement relative to baseline** (calibration reference point).
    - **Tremor amplitude**, based on short-time variability of the displacement.
    - **Drift**, based on slow trends away from the baseline.
    - **Fatigue index**, comparing early vs late segments of the test.

    These metrics are combined into a **single stability score (0–100)** for
    educational visualization.
    """
)

st.subheader("Limitations")
st.markdown(
    """
    - This tool has not been clinically validated.
    - Webcam resolution, lighting, and occlusions can introduce noise.
    - It does **not** replace formal clinical assessments of tremor or motor function.
    """
)

st.caption("Created as a final project for BME 3053C (Signals & Systems).")
