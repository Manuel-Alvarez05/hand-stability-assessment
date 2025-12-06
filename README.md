# 🩺 Hand Stability & Tremor Assessment Tool

A biomechanics + computer vision project that estimates **tremor, drift, and fatigue**
using **MediaPipe hand tracking and Streamlit**.

This tool simulates a **motor steadiness test** and is designed for biomechanics education,
signal processing exploration, and visualization—not medical diagnosis.

---

### 🔷 How It Works

| Step | Page | What Happens |
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

Open a terminal and run:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the URL printed in the terminal in your browser.

### Webcam permissions (browser-based)
- The app uses WebRTC to capture video directly from your browser. When prompted, click **Allow** for camera access.
- Works in GitHub-hosted/Codespaces environments because capture happens in the browser, not on the server.