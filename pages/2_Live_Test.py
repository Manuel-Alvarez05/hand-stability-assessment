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

    - Keep the same posture you used during calibration.
    - Try to hold your hand as still as you comfortably can.
    """
)

# Ensure calibration has been completed
baseline_positions = st.session_state.get("baseline_positions")
if not baseline_positions:
    st.error(
        "No baseline positions found. Please complete the calibration step first "
        "by visiting '1_Calibration' in the sidebar."
    )
    st.stop()

st.session_state.setdefault("test_complete", False)

# Layout
preview_col, controls_col = st.columns([2, 1])

with preview_col:
    preview_frame_placeholder = st.empty()
    status_placeholder = st.empty()

with controls_col:
    st.subheader("Webcam & Test Controls")

    # Start WebRTC stream (again, must run every rerun with a stable key)
    webrtc_ctx = mediapipe_utils.init_webrtc_stream("live-test-webrtc")

    # Debug info
    st.sidebar.subheader("Debug Info")
    if webrtc_ctx is not None:
        st.sidebar.write(f"WebRTC state: {getattr(webrtc_ctx, 'state', 'unknown')}")
        st.sidebar.write(
            "Video processor attached: "
            + ("✅" if getattr(webrtc_ctx, "video_processor", None) is not None else "❌")
        )
    else:
        st.sidebar.write("No WebRTC context yet.")

    # Preview before the test starts
    fingertips, frame_rgb = mediapipe_utils.get_latest_frame_and_fingertips(webrtc_ctx)

    if webrtc_ctx and getattr(webrtc_ctx, "state", None) and webrtc_ctx.state.playing and frame_rgb is not None:
        preview_frame_placeholder.image(frame_rgb, channels="RGB", caption="Live preview")
        status_placeholder.success("Camera connected. You can start the test when ready.")
    elif not (webrtc_ctx and getattr(webrtc_ctx, "state", None) and webrtc_ctx.state.playing):
        status_placeholder.info("Waiting for camera permission... click 'Allow' in your browser.")
    else:
        status_placeholder.info("Waiting for webcam frame...")

    st.markdown(
        f"Live test duration: **{config.TEST_DURATION_SECONDS} seconds**."
    )
    start_test = st.button("Start Live Test", type="primary")

if start_test:
    # Ensure the camera is actually playing
    if not (webrtc_ctx and getattr(webrtc_ctx, "state", None) and webrtc_ctx.state.playing):
        st.error("Camera is not connected. Please allow camera access and try again.")
        st.stop()

    status_placeholder.info("Recording... Hold your hand steady.")
    progress_bar = st.progress(0)
    timer_placeholder = st.empty()
    live_preview_placeholder = preview_frame_placeholder

    # Store time series: {finger: [(t, x, y), ...]}
    raw_time_series: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

    start_time = time.perf_counter()
    duration = config.TEST_DURATION_SECONDS

    while True:
        now = time.perf_counter()
        elapsed = now - start_time
        remaining = max(0.0, duration - elapsed)

        if elapsed >= duration:
            break

        # Update UI
        progress = min(1.0, elapsed / duration)
        progress_bar.progress(progress)
        timer_placeholder.text(f"Time remaining: {remaining:0.1f} s")

        # Get latest data
        fingertips, frame_rgb = mediapipe_utils.get_latest_frame_and_fingertips(webrtc_ctx)

        if frame_rgb is not None:
            live_preview_placeholder.image(frame_rgb, channels="RGB", caption="Recording...")

        if fingertips is not None:
            t_rel = elapsed
            for finger_name, (x, y) in fingertips.items():
                raw_time_series[finger_name].append((t_rel, x, y))

        time.sleep(0.03)

    # Save results to session state
    st.session_state["raw_time_series"] = raw_time_series
    st.session_state["test_complete"] = True

    status_placeholder.success("Live test complete! Data recorded successfully.")

    # Try to stop WebRTC gracefully
    if webrtc_ctx and getattr(webrtc_ctx, "state", None) and webrtc_ctx.state.playing:
        try:
            webrtc_ctx.stop()
        except Exception:
            pass

    # Navigate to results page
    try:
        time.sleep(0.5)
        st.switch_page("pages/3_Results.py")
    except AttributeError:
        # Older Streamlit versions don't provide `switch_page`
        st.info("Live test complete. Please open '3_Results' from the sidebar to view your results.")

# If the test was already completed earlier and the user just visited the page
if st.session_state.get("test_complete") and not start_test:
    status_placeholder.success("Live test already completed. You may proceed to the Results page.")
    st.caption("If you want to repeat the test, you can run it again by pressing the button.")
