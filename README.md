# Collision Analysis System GUI (`collision_gui.py`)

This project provides a Tkinter-based Emergency Dashboard for the Car Crash Analysis system. It integrates the backend **Roboflow** and **YOLO** inference logic from `analysis.py` into a user-friendly graphical interface. Features include live camera feed scanning, manual image uploads, visual bounding box overlays, and automated report logging.

---

## Python version and virtual environment

Use **Python 3.9, 3.10, 3.11, or 3.12** inside a dedicated virtual environment.

- **Avoid Python 3.13+** — `inference-sdk` and some dependency combos are unreliable or fail to install on very new Python releases.
- **Use the same interpreter for every command** (`python`, `pip`, and your IDE) so dependencies match.

  To set up a virtual environment (e.g., using Python 3.9):

  ```bash
  python3.9 -m venv .venv39 
  source .venv39/bin/activate   # Windows: .venv39\Scripts\activate
  python -m pip install --upgrade pip
  ```

---

## What to install

Install these **Python packages** into the activated venv. The GUI shares all the dependencies required by the backend `analysis.py` script:

| Package | Role |
|--------|------|
| `opencv-python` | Process video feeds, draw bounding boxes, and handle image color conversions |
| `ultralytics` | YOLO integration for vehicle counting inside crash zones |
| `inference-sdk` | Roboflow HTTP inference client to detect crash scenes |
| `python-dotenv` | Loads API keys and Model IDs from the `.env` file automatically |
| `Pillow` | Converts OpenCV BGR image arrays into Tkinter-compatible RGB `ImageTk` objects |

**One-liner Installation:**

```bash
python -m pip install opencv-python ultralytics inference-sdk python-dotenv pillow supervision
```

*(Note: Tkinter is included with standard Windows/macOS Python installations. Linux users may need to run `sudo apt-get install python3-tk` if not already installed).*

---

## Roboflow configuration (`.env`)

The GUI relies on a `.env` file to securely load credentials before launching. 

1. Create a `.env` file in the root directory.
2. Set **`ROBOFLOW_API_KEY`** (required).
3. Choose **one** inference style to define your model:

   - **Workflow:** `ROBOFLOW_WORKSPACE_NAME` and `ROBOFLOW_WORKFLOW_ID`

   - **Hosted model:** `ROBOFLOW_MODEL_ID` (e.g., `your-project/1`)

---

## How to run

From the project directory, with the venv **activated** and `.env` present:

```bash
python collision_gui.py
```

### Dashboard Features

- **Start Scan:** Accesses your local webcam (`cv2.VideoCapture(0)`) to run a continuous loop, analyzing a new frame every 1 second to prevent API rate-limiting. 
- **Stop Scan:** Safely releases the camera hardware and clears the simulation canvas.
- **Upload Image:** Pauses any active live feeds and opens a file dialog to analyze a single local image (`.jpg`, `.png`).
- **View Report:** Opens a pop-up window reading from `crash_report_log.txt`, detailing the history of detected crashes and vehicle counts.
- **Status Panel:** Dynamically updates to show the total number of vehicles involved in an active crash scene.

---

## Behavior summary

- **Unified Canvas Display:** Both live drone feeds and uploaded images are routed through a central renderer that automatically resizes frames to fit the Tkinter layout using `Image.Resampling.LANCZOS`.
- **Automated Logging:** Whenever a frame analysis returns a "CRASH DETECTED" string, the GUI automatically appends a timestamp, filename, and result message to a local `crash_report_log.txt` file.
- **Decoupled Backend:** The GUI imports `CrashAnalysisApp` from `analysis.py`. All API requests, YOLO overlap checks, and bounding box math are executed by the backend app before being returned to the GUI for rendering.

---

## Troubleshooting

| Issue | What to try |
|--------|-------------|
| `Missing ROBOFLOW_API_KEY` | Ensure your `.env` file is in the same directory as `collision_gui.py` and correctly formatted. |
| GUI Freezes / Becomes Unresponsive | Ensure `update_scan()` is using `self.root.after()` rather than a `while True` loop, which blocks Tkinter's main event thread. |
| Cannot open camera feed | Check system privacy settings to ensure Python has permission to access the webcam, or verify the camera index in `cv2.VideoCapture(0)`. |
| `AttributeError: module 'cv2' has no attribute...` | A corrupt OpenCV installation. Run `pip uninstall opencv-python` followed by `pip install opencv-python`. |