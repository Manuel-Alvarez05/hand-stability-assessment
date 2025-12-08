"""
Browser-based webcam capture for MediaPipe Hands using Streamlit WebRTC.

This module processes frames sent from the browser (with user permission)
rather than relying on server-side webcams, which are typically unavailable
in GitHub Codespaces or other remote deployments.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import av
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

from core import config


# -----------------------------
# MediaPipe configuration
# -----------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark indices for fingertip points (MediaPipe Hands convention)
FINGERTIP_INDICES: Dict[str, int] = {
    "THUMB": 4,
    "INDEX": 8,
    "MIDDLE": 12,
    # You can add "RING": 16, "PINKY": 20 if you decide to track more fingers later
}

# Sanity-check configuration
_unknown = set(config.FINGERS_TO_TRACK) - set(FINGERTIP_INDICES)
if _unknown:
    raise ValueError(
        f"Unknown fingers in config.FINGERS_TO_TRACK: {_unknown}. "
        f"Supported keys: {list(FINGERTIP_INDICES.keys())}"
    )


def _extract_fingertip_coords(
    landmarks: "list[mp.framework.formats.landmark_pb2.NormalizedLandmark]",
) -> Dict[str, Tuple[float, float]]:
    """
    Extract (x, y) normalized coordinates for configured fingertips from a landmark list.

    Coordinates are in [0, 1] relative to image width/height, as returned by MediaPipe.
    """
    fingertip_coords: Dict[str, Tuple[float, float]] = {}

    for finger_name in config.FINGERS_TO_TRACK:
        idx = FINGERTIP_INDICES.get(finger_name)
        if idx is None or idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        fingertip_coords[finger_name] = (lm.x, lm.y)

    return fingertip_coords


class MediaPipeHandProcessor(VideoProcessorBase):
    """
    WebRTC video processor that runs MediaPipe Hands on each incoming frame.

    It:
    - Receives frames from the browser.
    - Runs MediaPipe Hands.
    - Draws landmarks on the frame.
    - Stores the latest RGB frame and fingertip coordinates under a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.latest_frame_rgb: Optional[np.ndarray] = None
        self.latest_fingertips: Optional[Dict[str, Tuple[float, float]]] = None

    def __del__(self) -> None:
        # Best-effort cleanup; ignore errors during interpreter shutdown
        try:
            self.hands.close()
        except Exception:
            pass

    def get_latest(self) -> Tuple[Optional[Dict[str, Tuple[float, float]]], Optional[np.ndarray]]:
        """
        Return the latest (fingertips, frame_rgb) pair.

        - fingertips: dict[finger_name -> (x, y)] in normalized coordinates.
        - frame_rgb:  H x W x 3 uint8 RGB image suitable for st.image(...).
        """
        with self._lock:
            if self.latest_frame_rgb is None:
                return None, None
            # Return a copy so callers don't accidentally mutate internal state
            return self.latest_fingertips, self.latest_frame_rgb.copy()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Process an incoming frame:
        - Convert to RGB
        - Run MediaPipe Hands
        - Draw landmarks
        - Update latest frame + fingertips

        This method is wrapped in a fail-safe try/except so that if MediaPipe or
        OpenCV throws, the stream keeps running and we still show the raw frame
        instead of killing the WebRTC connection (which would look like a
        permanently black video saying "waiting for camera").
        """
        try:
            # Convert incoming frame to BGR for OpenCV
            frame_bgr = frame.to_ndarray(format="bgr24")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            results = self.hands.process(frame_rgb)

            fingertips: Optional[Dict[str, Tuple[float, float]]] = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                fingertips = _extract_fingertip_coords(hand_landmarks.landmark)

                # Draw landmarks & connections for user feedback
                mp_drawing.draw_landmarks(
                    frame_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            # Store latest under a lock
            with self._lock:
                self.latest_frame_rgb = frame_rgb
                self.latest_fingertips = fingertips

            # Convert back to BGR for sending
            annotated_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

        except Exception as e:
            # Fail-safe: if anything goes wrong, just return the original frame so the
            # WebRTC stream doesn't die. Optionally log in debug mode.
            if config.DEBUG_MODE:
                print(f"[MediaPipeHandProcessor] Error in recv: {e!r}")

            try:
                frame_bgr = frame.to_ndarray(format="bgr24")
                return av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
            except Exception:
                # If even this fails, re-raise; at that point something is seriously off.
                raise


def init_webrtc_stream(key: str):
    """
    Start/return a WebRTC streamer that prompts for browser camera access.

    Parameters
    ----------
    key:
        A stable string that uniquely identifies the component on the page.
        Use different keys for Calibration and Live Test, but keep them fixed
        across reruns (e.g. "calibration-webrtc", "live-test-webrtc").
    """
    return webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=MediaPipeHandProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def get_latest_frame_and_fingertips(
    webrtc_ctx,
) -> Tuple[Optional[Dict[str, Tuple[float, float]]], Optional[np.ndarray]]:
    """
    Fetch the latest processed frame and fingertips from a WebRTC context.

    Returns
    -------
    (fingertips, frame_rgb)
        fingertips: dict or None
        frame_rgb:  RGB numpy array or None
    """
    if webrtc_ctx is None or webrtc_ctx.video_processor is None:
        return None, None
    return webrtc_ctx.video_processor.get_latest()
