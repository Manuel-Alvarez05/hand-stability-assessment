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

On Ubuntu/Debian, ensure OpenCV runtime libraries are present (required by `cv2` even in headless builds):

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

If you are using GitHub Codespaces or a dev container, run the apt commands once after the container is created. This resolves native module bootstrapping errors when importing `cv2`.
## Troubleshooting

- Camera won’t start after revisiting pages:
	- Use the "Reset Camera" button on Calibration or Live Test to release and reinitialize the WebRTC stream.
	- Make sure your browser granted camera permission to the app.
- `ImportError` during `cv2` import:
	- Install system libs: `sudo apt-get install -y libgl1 libglib2.0-0`
	- Then verify: `python -c "import cv2; print(cv2.__version__)"`


Then open the URL printed in the terminal in your browser.

### Webcam permissions (browser-based)
- The app uses WebRTC to capture video directly from your browser. When prompted, click **Allow** for camera access.
- Works in GitHub-hosted/Codespaces environments because capture happens in the browser, not on the server.