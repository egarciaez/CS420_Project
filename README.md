# Drone Security Perimeter Simulator

This project simulates an autonomous drone security response to a vehicle incident. Using computer vision and a pre-trained car detection model, the script identifies a "crash" site and animates a drone patrolling a perimeter to "secure" the scene.

## 🚀 Overview

The simulation follows a four-stage pipeline:
1.  **Object Detection:** Uses `inference` (Roboflow) to locate cars/crashes in a top-down image.
2.  **Path Calculation:** Generates a rectangular "safety perimeter" around detected objects with calculated padding.
3.  **Alpha Blending:** Dynamically overlays a transparent drone `.png` using custom alpha-channel math for a realistic look.
4.  **Animated Patrol:** The drone moves along the perimeter, pausing at specific intervals to simulate a "task" (like placing safety cones).

---

## 🛠️ Requirements

* **Python 3.x**
* **OpenCV (`cv2`)**: For image processing and UI.
* **NumPy**: For coordinate and array calculations.
* **Inference**: To run the Roboflow model.

```bash
pip install opencv-python numpy inference
