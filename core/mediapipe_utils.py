# core/mediapipe_utils.py

"""
This module will contain helper functions to:
- Initialize MediaPipe Hands.
- Capture frames from the webcam.
- Extract fingertip landmark coordinates for THUMB, INDEX, MIDDLE.
"""

def init_mediapipe_hands():
    """
    TODO (AI / TEAM):
    - Import mediapipe.
    - Create and return a Hands object configured for real-time video.
    """
    raise NotImplementedError

def capture_frame_and_landmarks(hands_context):
    """
    TODO (AI / TEAM):
    - Capture a single frame from the default webcam (OpenCV).
    - Run MediaPipe hand detection on the frame.
    - If a hand is detected:
        * Extract fingertip coordinates for THUMB, INDEX, MIDDLE.
        * Return the frame (for display) and a dict like:
          {
              "THUMB": (x_thumb, y_thumb),
              "INDEX": (x_index, y_index),
              "MIDDLE": (x_middle, y_middle),
          }
      - If no hand is detected, return None or an empty dict.
    """
    raise NotImplementedError
