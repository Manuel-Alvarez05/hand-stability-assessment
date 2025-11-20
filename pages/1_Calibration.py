import streamlit as st
from core import config
# from core import mediapipe_utils   # TO BE IMPLEMENTED BY TEAM/AI

st.set_page_config(page_title="Calibration", page_icon="🎯", layout="wide")

st.title("Step 1: Calibration")

st.markdown(
    """
    In this step, we capture a **baseline reference position** for your fingertips.

    **Instructions:**
    - Sit at a comfortable distance from your webcam.
    - Raise your **dominant hand** so it is fully visible.
    - Extend your **thumb, index, and middle fingers**.
    - Try to hold your hand as steady as possible inside the on-screen guide.
    """
)

st.warning("This tool is for educational purposes only and does **not** diagnose any condition.")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Webcam & Hand Preview")
    # TODO (AI / TEAM):
    # - Use Streamlit's camera input or OpenCV video stream.
    # - Overlay MediaPipe hand landmarks (thumb, index, middle) on the video.
    # - Provide a simple visual target region (e.g., a box) for the user to align their hand.
    #
    # PSEUDOCODE:
    # frame = capture_frame_from_webcam()
    # landmarks = detect_hand_landmarks(frame)
    # draw_visual_guides(frame, landmarks)
    # st.image(frame) OR st.camera_input(...)
    st.info("Webcam preview and landmark overlay will appear here.")

with col2:
    st.subheader("Calibration Control")

    st.markdown(
        f"""
        When you are ready and your hand is steady, click the button below.
        We will record **{config.CALIBRATION_DURATION_SECONDS} seconds** of baseline data.
        """
    )

    if st.button("▶ Run Calibration"):
        # TODO (AI / TEAM):
        # - For the next CALIBRATION_DURATION_SECONDS, continuously:
        #   * Capture hand landmarks (thumb, index, middle).
        #   * Store their positions (e.g., x, y pixel coords or normalized coords).
        # - Compute the **average position** for each finger over this window.
        # - Save these baseline positions into st.session_state["baseline_positions"].
        #
        # Example schema for baseline:
        # st.session_state["baseline_positions"] = {
        #     "THUMB": (x_thumb_mean, y_thumb_mean),
        #     "INDEX": (x_index_mean, y_index_mean),
        #     "MIDDLE": (x_middle_mean, y_middle_mean),
        # }
        #
        # - Display a success message when calibration finishes.
        st.session_state["calibration_complete"] = True
        st.success("Calibration started. (Logic to capture baseline will be implemented here.)")

if st.session_state.get("calibration_complete"):
    st.success("Calibration complete! You can move on to the Live Test page.")
    st.caption("Use the sidebar to go to '2_Live_Test'.")
