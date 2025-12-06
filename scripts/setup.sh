#!/usr/bin/env bash
set -euo pipefail

# Install Python deps
pip install -r requirements.txt

# Install system libs needed by OpenCV (cv2)
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y libgl1 libglib2.0-0 || true
fi

python - <<'PY'
try:
    import cv2
    print("cv2 OK:", cv2.__version__)
except Exception as e:
    print("cv2 import failed:", e)
    print("If on Ubuntu/Debian, run: sudo apt-get install -y libgl1 libglib2.0-0")
PY

echo "Setup complete. Run: streamlit run app.py"
