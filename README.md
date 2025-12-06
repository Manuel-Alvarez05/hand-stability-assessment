# 🩺 Hand Stability & Tremor Assessment Tool

A biomechanics + computer vision project that estimates **tremor, drift, and fatigue**
using **MediaPipe hand tracking and Streamlit**.

This tool simulates a **motor steadiness test** and is designed for biomechanics education,
signal processing exploration, and visualization—not medical diagnosis.

---

### 🔷 How It Works

| Step | Page | What Happens |
|---|---|---|
| 1️⃣ Calibration | `1_Calibration.py` | 3-second recording establishes baseline fingertip position |
| 2️⃣ Live Test | `2_Live_Test.py` | 30-sec MediaPipe tracking of THUMB, INDEX, MIDDLE |
| 3️⃣ Results | `3_Results.py` | Tremor, drift, fatigue + stability score visualized |

Metrics are computed as:

| Metric | Meaning |
|---|---|
| Tremor (RMS) | Magnitude of displacement from baseline |
| Drift | Start→end posture change |
| Fatigue Index | Late RMS / Early RMS |
| Stability Score (0–100) | Higher = more stable |

---

### Installation / Run Instructions

### Open git bash terminal and run the following commands:
```bash
pip install -r requirements.txt
python -m streamlit run app.py


### Terminal will provide a local host url, from there open that link in a web browser.