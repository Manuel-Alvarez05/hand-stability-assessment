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

    - Raise your hand in front of the camera.
    - Extend your thumb, index, and middle fingers.
    - Try to hold as still as possible while we measure your baseline.
    """
)

# Initialize session-state flags if missing
st.session_state.setdefault("calibration_complete", False)
st.session_state.setdefault("baseline_positions", {})

# Create main layout
preview_col, controls_col = st.columns([2, 1])

with preview_col:
    preview_frame_placeholder = st.empty()
    status_placeholder = st.empty()

with controls_col:
    st.subheader("Webcam & Calibration Controls")

    # Start WebRTC stream (this must run on every rerun with a stable key)
    webrtc_ctx = mediapipe_utils.init_webrtc_stream("calibration-webrtc")

    # Optional debug info in the sidebar
    st.sidebar.subheader("Debug Info")
    if webrtc_ctx is not None:
        st.sidebar.write(f"WebRTC state: {getattr(webrtc_ctx, 'state', 'unknown')}")
        st.sidebar.write(
            "Video processor attached: "
            + ("✅" if getattr(webrtc_ctx, "video_processor", None) is not None else "❌")
        )
    else:
        st.sidebar.write("No WebRTC context yet.")

    # Live preview (non-blocking)
    fingertips, frame_rgb = mediapipe_utils.get_latest_frame_and_fingertips(webrtc_ctx)

    if webrtc_ctx and getattr(webrtc_ctx, "state", None) and webrtc_ctx.state.playing and frame_rgb is not None:
        preview_frame_placeholder.image(frame_rgb, channels="RGB", caption="Live preview")
        status_placeholder.success("Camera connected. Adjust your hand position.")
    elif not (webrtc_ctx and getattr(webrtc_ctx, "state", None) and webrtc_ctx.state.playing):
        status_placeholder.info("Waiting for camera permission... click 'Allow' in your browser.")
    else:
        status_placeholder.info("Waiting for webcam frame...")

    st.markdown(
        f"Calibration will record **{config.CALIBRATION_DURATION_SECONDS} seconds** of data once you start."
    )
    start_calibration = st.button("Run Calibration", type="primary")


if start_calibration:
    # Ensure the camera is actually playing before we start collecting
    if not (webrtc_ctx and getattr(webrtc_ctx, "state", None) and webrtc_ctx.state.playing):
        st.error("Camera is not connected. Please allow camera access and try again.")
        st.stop()

    status_placeholder.info("Calibrating... Hold your hand as still as possible.")
    progress_bar = st.progress(0)
    timer_placeholder = st.empty()
    live_preview_placeholder = preview_frame_placeholder  # reuse

    samples: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    start_time = time.perf_counter()
    duration = config.CALIBRATION_DURATION_SECONDS

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

        # Grab latest frame and fingertips
        fingertips, frame_rgb = mediapipe_utils.get_latest_frame_and_fingertips(webrtc_ctx)

        if frame_rgb is not None:
            live_preview_placeholder.image(frame_rgb, channels="RGB", caption="Calibrating...")

        if fingertips is not None:
            for finger_name, (x, y) in fingertips.items():
                samples[finger_name].append((x, y))

        # Small sleep to avoid hammering the CPU
        time.sleep(0.03)

    # Compute baseline positions as the average of the collected samples
    baseline_positions: Dict[str, Tuple[float, float]] = {}
    for finger_name, coords in samples.items():
        if not coords:
            continue
        xs, ys = zip(*coords)
        baseline_positions[finger_name] = (float(np.mean(xs)), float(np.mean(ys)))

    if not baseline_positions:
        st.error(
            "No fingertip data was captured during calibration. "
            "Please make sure your hand is clearly visible and try again."
        )
        st.stop()

    # Save to session state
    st.session_state["baseline_positions"] = baseline_positions
    st.session_state["calibration_complete"] = True

    status_placeholder.success("Calibration complete! Baseline positions captured.")
    st.write("Baseline fingertip positions:", baseline_positions)

    # Try to stop the WebRTC stream gracefully
    if webrtc_ctx and getattr(webrtc_ctx, "state", None) and webrtc_ctx.state.playing:
        try:
            webrtc_ctx.stop()
        except Exception:
            pass

    # Navigate to the next page if available
    try:
        time.sleep(0.5)
        st.switch_page("pages/2_Live_Test.py")
    except AttributeError:
        # Older Streamlit versions don't provide `switch_page`; show a friendly hint instead
        st.info("Calibration complete. Please open '2_Live_Test' from the sidebar to continue.")

# If calibration was already completed in a prior run, show a friendly reminder
if st.session_state.get("calibration_complete") and not start_calibration:
    st.success("Calibration already completed. You may proceed to the Live Test page.")
    st.caption("If needed, you can recalibrate by pressing the button again.")

