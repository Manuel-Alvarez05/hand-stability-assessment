# core/config.py

# ---- TIMING ----
CALIBRATION_DURATION_SECONDS = 3
TEST_DURATION_SECONDS = 30

# ---- FINGERS WE TRACK ----
FINGERS_TO_TRACK = ["THUMB", "INDEX", "MIDDLE"]

# ---- COLOR MAPPING FOR FINGERS (for plots/UI) ----
FINGER_COLORS = {
    "THUMB": "#1E88E5",   # blue
    "INDEX": "#43A047",   # green
    "MIDDLE": "#E53935",  # red
}

# ---- STABILITY SCORE WEIGHTS (for AI to use in scoring.py) ----
WEIGHT_TREMOR = 0.4
WEIGHT_DRIFT = 0.3
WEIGHT_FATIGUE = 0.3

# NOTE TO AI / TEAM:
# - Use these constants throughout the app.
# - Do NOT hardcode durations or colors in other files.
