# Car crash analysis (`analysis.py`)

This project runs a **Roboflow** workflow or hosted model to detect crashes, with an optional **YOLO** pass for vehicles when Roboflow does not report a confident crash (useful for drone or wide shots). Results are drawn on the image and shown in an **OpenCV** window.

---

## Python version and virtual environment

Use **Python 3.9, 3.10, 3.11, or 3.12** inside a dedicated virtual environment.

- **Avoid Python 3.13+** — **`inference-sdk`** and some dependency combos are unreliable or fail to install on very new Python releases.
- **Use the same interpreter for every command** (`python`, `pip`, and your IDE) so dependencies match.

  This project has been used with **Python 3.9** so reccomend using it (e.g. a venv named `.venv39`). You can use **3.10–3.12** the same way; just swap the command and folder name:

  ```bash
  python3 -m venv .venv39 
  source .venv39/bin/activate   # Windows: .venv39\Scripts\activate
  python -m pip install --upgrade pip
  ```

- If you see **NumPy / PyTorch** binary errors after installing, try pinning NumPy 1.x before or after installing the rest:

  ```bash
  python -m pip install "numpy<2"
  ```

---

## What to install

Install these **Python packages** into the activated venv:

| Package | Role |
|--------|------|
| `opencv-python` | Read images, draw boxes, show the result window |
| `ultralytics` | YOLO (optional fallback / overlap); pulls **PyTorch** |
| `inference-sdk` | Roboflow HTTP inference client |
| `python-dotenv` | Load `.env` automatically (optional but recommended) |
| `Pillow` | Clearer antialiased overlay text (recommended) |

Example one-liner:

```bash
python -m pip install opencv-python ultralytics inference-sdk python-dotenv pillow
```

**First YOLO run:** Ultralytics will download default weights (e.g. `yolov8n.pt`) on first use unless you set `YOLO_WEIGHTS` to another file path.

**System requirements:**

- **macOS / Windows / Linux desktop** with a display if you use the interactive file picker and OpenCV window. On Linux, a running X11 or Wayland session is expected for GUI features.
- **macOS:** The default image picker uses **AppleScript / Finder** (`osascript`), not Tk, to avoid broken Tk builds on some Python installs.

---

## Roboflow configuration (`.env`)

1. Copy the template and add your key:

   ```bash
   cp .env.example .env
   ```

2. Set **`ROBOFLOW_API_KEY`** (required).

3. Choose **one** inference style:

   - **Workflow (recommended if you deploy from Roboflow as a workflow):**  
     `ROBOFLOW_WORKSPACE_NAME` and `ROBOFLOW_WORKFLOW_ID`

   - **Hosted model:**  
     `ROBOFLOW_MODEL_ID` (e.g. `your-project/1`)

   If both workflow and model variables are set, the app uses the **workflow** path when workspace + workflow id are present.

See `.env.example` for optional tuning (YOLO fallback, confidence thresholds, overlay fonts, etc.).

---

## How to run

From the project directory, with the venv **activated** and `.env` present:

```bash
python analysis.py
```

- **Interactive mode (default):** A file picker opens; after each result window, pick another image until you quit from the window.
- **Single pick then exit:**

  ```bash
  python analysis.py --once
  ```

- **Skip the picker** (pass a path directly):

  ```bash
  python analysis.py --image /path/to/photo.jpg
  ```

### Result window

- The annotated image opens in an OpenCV window titled **“Detection Result”**.
- Press **Q** or **Esc** to quit the picker loop; any other key continues to the next image (when not using `--once`).

---

## Behavior summary

- **Crash detection** comes from Roboflow predictions (class name heuristics + `CRASH_MIN_CONFIDENCE`, default `0.35`).
- **YOLO fallback** (`YOLO_FALLBACK`, default on): If there is no confident crash, the script runs YOLO for **car / truck / bus** (and optionally motorcycle) and draws green boxes, with a short status label.
- **Overlay text** uses Pillow + a system font when available; set `OVERLAY_USE_CV2_TEXT=1` to force OpenCV’s built-in font.
- **Debug:** Set `DEBUG_ROBOFLOW=1` for extra terminal output (including empty Roboflow responses).

---

## Troubleshooting

| Issue | What to try |
|--------|-------------|
| `Missing ROBOFLOW_API_KEY` | Add the key to `.env` or export it in the shell. |
| `ModuleNotFoundError` (e.g. `cv2`, `inference_sdk`) | Activate the correct venv; run `pip install` with **`python -m pip`** tied to that venv. |
| `inference-sdk` install fails | Use Python **3.9–3.12**; upgrade `pip` and retry. |
| NumPy / Torch errors | `python -m pip install "numpy<2"` then reinstall stack if needed. |
| OpenCV window or picker does not appear (Linux) | Ensure `DISPLAY` or `WAYLAND_DISPLAY` is set; run from a desktop session. |
| macOS Tk crashes when picking files | Keep **`MACOS_IMAGE_PICKER`** unset or `osascript` (default). |

---

## Project layout

- **`analysis.py`** — Main entry: Roboflow + optional YOLO, overlays, interactive or CLI image path.
- **`.env.example`** — Documented environment variables; copy to **`.env`** (keep `.env` private; do not commit API keys).
