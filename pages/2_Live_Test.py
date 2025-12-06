import time
from collections import defaultdict
from typing import Dict, List, Tuple

import streamlit as st

from core import config
from core import mediapipe_utils


st.title("Step 2: Live Stability Test")

st.markdown(
    f"""
    In this step, we record **{config.TEST_DURATION_SECONDS} seconds** of fingertip motion
    while you hold your hand as steady as possible.

    **Instructions:**
    - Keep the **same hand and position** you used during calibration.
    - Extend your **thumb, index, and middle fingers**.
    - Try to hold your hand as still as you can for the entire test duration.
    """
)

st.warning(
    "This is an educational tool and does **not** provide a clinical diagnosis."
)

st.divider()

# -----------------------------------
# Check that calibration has been completed
# -----------------------------------
baseline_positions = st.session_state.get("baseline_positions", {})

if not baseline_positions:
    st.error(
        "Baseline positions not found. Please complete **Step 1: Calibration** before running the test."
    )
    st.stop()

# -----------------------------------
# Initialize / reuse browser webcam stream (prompts for permission)
# Support restarting via a session-scoped key counter
# -----------------------------------
if "webrtc_key_live_counter" not in st.session_state:
    st.session_state["webrtc_key_live_counter"] = 0

webrtc_key = f"live-test-webrtc-{st.session_state['webrtc_key_live_counter']}"
webrtc_ctx = mediapipe_utils.init_webrtc_stream(webrtc_key)

# Flag to track whether the test has been run successfully
if "test_complete" not in st.session_state:
    st.session_state["test_complete"] = False

# Initialize raw_time_series if not already
if "raw_time_series" not in st.session_state:
    st.session_state["raw_time_series"] = {
        finger: [] for finger in config.FINGERS_TO_TRACK
    }

# -----------------------------------
# Layout: video + progress on left, controls on right
# -----------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Webcam & Tracking View")
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    progress_placeholder = st.empty()

with col2:
    st.subheader("Test Control")
    st.markdown(
        f"""
        When you press the button below, we will record **{config.TEST_DURATION_SECONDS} seconds**
        of fingertip motion.

        Try to hold your hand as steady as possible during the entire test.
        """
    )
    start_test = st.button("▶ Start Live Test")
    reset_camera = st.button("🔄 Reset Camera")

    if reset_camera:
        # Stop current stream and force a fresh component instance on rerun
        mediapipe_utils.stop_webrtc_stream(webrtc_ctx)
        st.session_state["webrtc_key_live_counter"] += 1
        # Clear flags/data so a new run behaves predictably
        st.session_state["test_complete"] = False
        st.session_state["raw_time_series"] = {finger: [] for finger in config.FINGERS_TO_TRACK}
        st.rerun()


# -----------------------------------
# Live preview (single frame per rerun)
# -----------------------------------
if not st.session_state.get("test_complete"):
    fingertips, frame_rgb = mediapipe_utils.get_latest_frame_and_fingertips(webrtc_ctx)
    if webrtc_ctx and webrtc_ctx.state.playing and frame_rgb is not None:
        video_placeholder.image(
            frame_rgb,
            caption="Live preview (landmarks shown if detected)",
            channels="RGB",
        )
    elif not (webrtc_ctx and webrtc_ctx.state.playing):
        status_placeholder.info("Waiting for camera permission... Click 'Allow' in your browser.")
    else:
        status_placeholder.info("Waiting for webcam frame... Ensure your camera is enabled.")


# -----------------------------------
# Perform the timed test on button click
# -----------------------------------
if start_test:
    if not (webrtc_ctx and webrtc_ctx.state.playing):
        status_placeholder.error("Camera stream is not running. Please allow camera access and try again.")
        st.stop()

    st.session_state["test_complete"] = False
    status_placeholder.info("Test in progress... Hold your hand steady.")
    progress_bar = progress_placeholder.progress(0)

    # Reset raw_time_series dict
    raw_time_series: Dict[str, List[Tuple[float, float, float]]] = {
        finger: [] for finger in config.FINGERS_TO_TRACK
    }

    start_time = time.time()
    duration = config.TEST_DURATION_SECONDS

    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration:
            break

        # Capture current frame and fingertip positions
        fingertip_positions, frame_rgb = mediapipe_utils.get_latest_frame_and_fingertips(webrtc_ctx)

        if frame_rgb is not None:
            video_placeholder.image(
                frame_rgb,
                caption="Recording in progress... Keep your hand as steady as possible.",
                channels="RGB",
            )

        # Save fingertip positions if detected
        if fingertip_positions:
            for finger_name, (x_norm, y_norm) in fingertip_positions.items():
                if finger_name in config.FINGERS_TO_TRACK:
                    # Store a tuple (t, x, y) with time relative to test start
                    raw_time_series[finger_name].append((float(elapsed), float(x_norm), float(y_norm)))

        # Update progress bar (0–100)
        progress = min(1.0, elapsed / duration)
        progress_bar.progress(int(progress * 100))

        # Slight delay to approximate ~30 FPS and avoid CPU overload
        time.sleep(1 / 30.0)

    # Save collected data into session_state
    st.session_state["raw_time_series"] = raw_time_series
    st.session_state["test_complete"] = True

    status_placeholder.success("Live test complete! Data has been recorded.")
    progress_placeholder.empty()  # Clear progress bar

    st.caption(
        "You can now proceed to **Step 3: Results & Interpretation** using the navigation menu."
    )

    # Optionally stop the camera to avoid browser/device locking if user navigates away
    mediapipe_utils.stop_webrtc_stream(webrtc_ctx)

# If the test was already completed earlier and user just visited the page
if st.session_state.get("test_complete") and not start_test:
    status_placeholder.success("Live test already completed. You may proceed to the Results page.")
    st.caption("If you want to repeat the test, you can run it again by pressing the button.")
