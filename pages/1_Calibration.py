import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

from core import config
from core import mediapipe_utils


st.title("Step 1: Calibration")

st.markdown(
    """
    In this step, we capture a **baseline reference position** for your fingertips.

    **Instructions:**
    - Sit at a comfortable distance from your webcam.
    - Raise your **dominant hand** so it is fully visible.
    - Extend your **thumb, index, and middle fingers**.
    - Try to hold your hand as steady as possible inside the on-screen view.
    """
)

st.warning(
    "This tool is for educational purposes only and does **not** provide a medical diagnosis."
)

st.divider()

# -----------------------------------
# Initialize / reuse browser webcam stream (prompts for permission)
# -----------------------------------
webrtc_ctx = mediapipe_utils.init_webrtc_stream("calibration-webrtc")

if "calibration_complete" not in st.session_state:
    st.session_state["calibration_complete"] = False

if "baseline_positions" not in st.session_state:
    st.session_state["baseline_positions"] = {}


# -----------------------------------
# Layout: video on left, controls on right
# -----------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Webcam & Hand Preview")
    preview_frame_placeholder = st.empty()
    status_placeholder = st.empty()

with col2:
    st.subheader("Calibration Control")
    st.markdown(
        f"""
        When you are ready and your hand is steady, click the button below.
        We will record **{config.CALIBRATION_DURATION_SECONDS} seconds** of baseline fingertip data.
        """
    )
    start_calibration = st.button("▶ Run Calibration")


# -----------------------------------
# Continuous preview before/after calibration
# -----------------------------------
fingertips, frame_rgb = mediapipe_utils.get_latest_frame_and_fingertips(webrtc_ctx)
if webrtc_ctx and webrtc_ctx.state.playing and frame_rgb is not None:
    preview_frame_placeholder.image(
        frame_rgb,
        caption="Webcam preview (with hand landmarks, if detected)",
        channels="RGB",
    )
elif not (webrtc_ctx and webrtc_ctx.state.playing):
    status_placeholder.info("Waiting for camera permission... Click 'Allow' in your browser.")
else:
    status_placeholder.info("Waiting for webcam frame... Make sure your camera is enabled.")


# -----------------------------------
# Perform calibration capture when button is clicked
# -----------------------------------
if start_calibration:
    if not (webrtc_ctx and webrtc_ctx.state.playing):
        status_placeholder.error("Camera stream is not running. Please allow camera access and try again.")
        st.stop()

    st.session_state["calibration_complete"] = False
    status_placeholder.info("Calibration in progress... Hold your hand steady.")

    # Use a dict of finger -> list of (x, y) pairs
    samples: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    start_time = time.time()
    duration = config.CALIBRATION_DURATION_SECONDS

    # Capture frames for the specified duration
    while time.time() - start_time < duration:
        fingertip_positions, frame_rgb = mediapipe_utils.get_latest_frame_and_fingertips(webrtc_ctx)

        if frame_rgb is not None:
            preview_frame_placeholder.image(
                frame_rgb,
                caption="Calibration in progress... Keep your hand steady.",
                channels="RGB",
            )

        # If landmarks detected, store them
        if fingertip_positions:
            for finger_name, (x, y) in fingertip_positions.items():
                if finger_name in config.FINGERS_TO_TRACK:
                    samples[finger_name].append((x, y))

        # Small sleep to avoid hammering the CPU (approx 30 FPS)
        time.sleep(1 / 30.0)

    # Compute mean position per finger
    baseline_positions: Dict[str, Tuple[float, float]] = {}

    for finger_name in config.FINGERS_TO_TRACK:
        coords = samples.get(finger_name, [])
        if len(coords) == 0:
            continue  # no data for this finger

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        baseline_positions[finger_name] = (float(np.mean(xs)), float(np.mean(ys)))

    if len(baseline_positions) == 0:
        status_placeholder.error(
            "Calibration failed: no hand landmarks were detected. "
            "Please adjust your lighting and hand position, then try again."
        )
    else:
        st.session_state["baseline_positions"] = baseline_positions
        st.session_state["calibration_complete"] = True
        status_placeholder.success("Calibration complete! Baseline positions have been saved.")

        st.caption(
            "You can now move on to **Step 2: Live Test** using the navigation menu on the left."
        )

# If calibration was already completed in a prior run, show a friendly reminder
if st.session_state.get("calibration_complete") and not start_calibration:
    st.success("Calibration already completed. You may proceed to the Live Test page.")
    st.caption("If needed, you can recalibrate by pressing the button again.")
