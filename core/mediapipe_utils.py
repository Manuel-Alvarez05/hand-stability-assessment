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
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from core import config


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Mapping from finger names to MediaPipe landmark indices
FINGERTIP_INDICES = {"THUMB": 4, "INDEX": 8, "MIDDLE": 12}


def _extract_fingertip_coords(landmarks) -> Dict[str, Tuple[float, float]]:
    """Convert MediaPipe landmarks into normalized fingertip coords."""
    fingertip_positions: Dict[str, Tuple[float, float]] = {}

    for finger_name in config.FINGERS_TO_TRACK:
        mp_index = FINGERTIP_INDICES.get(finger_name)
        if mp_index is None:
            continue

        lm = landmarks[mp_index]
        x_norm = max(0.0, min(1.0, float(lm.x)))
        y_norm = max(0.0, min(1.0, float(lm.y)))
        fingertip_positions[finger_name] = (x_norm, y_norm)

    return fingertip_positions


class MediaPipeHandProcessor(VideoProcessorBase):
    """WebRTC video processor that runs MediaPipe Hands per frame."""

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

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert incoming frame to BGR for OpenCV
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = self.hands.process(frame_rgb)

        fingertips: Optional[Dict[str, Tuple[float, float]]] = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            fingertips = _extract_fingertip_coords(hand_landmarks.landmark)

            mp_drawing.draw_landmarks(
                frame_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        with self._lock:
            self.latest_frame_rgb = frame_rgb
            self.latest_fingertips = fingertips

        # Convert back to BGR for returning to client
        annotated_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

    def get_latest(self) -> Tuple[Optional[Dict[str, Tuple[float, float]]], Optional[np.ndarray]]:
        """Thread-safe retrieval of the most recent landmarks and frame."""
        with self._lock:
            if self.latest_frame_rgb is None:
                return None, None
            return self.latest_fingertips, self.latest_frame_rgb.copy()


def init_webrtc_stream(key: str):
    """Start/return a WebRTC streamer that prompts for browser camera access."""
    return webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=MediaPipeHandProcessor,
        media_stream_constraints={
            "video": {"width": {"min": 640, "ideal": 1280, "max": 1920}},
            "audio": False
        },
        async_processing=True,
        rtc_configuration={"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]}
        ]},
    )


def get_latest_frame_and_fingertips(webrtc_ctx) -> Tuple[Optional[Dict[str, Tuple[float, float]]], Optional[np.ndarray]]:
    """Fetch the latest processed frame and fingertips from a WebRTC context."""
    if webrtc_ctx is None or webrtc_ctx.video_processor is None:
        return None, None
    return webrtc_ctx.video_processor.get_latest()
