# main script for our crash detection demo (opencv + yolo + roboflow)
import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient

try:
    # Optional for .env file.
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

def _env_bool(key: str, default: bool) -> bool:
    # Read an env var as boolean (1/true/yes etc); if missing or weird, return default.
    raw = os.environ.get(key, "").strip().lower()
    if raw == "":
        return default
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return default


def _env_float(key: str, default: float) -> float:
    # Read an env var as float; empty or invalid string -> use default.
    raw = os.environ.get(key, "").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    # Read an env var as int; empty or invalid string -> use default.
    raw = os.environ.get(key, "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _pick_image_macos() -> str:
    # For macOS opens a Finder "choose file" dialog via AppleScript (avoids flaky Tk on some Python builds).
    script = (
        'set p to POSIX path of (choose file with prompt '
        '"Select an image" of type {"public.jpeg", "public.png", "public.tiff"})'
    )
    try:
        out = subprocess.check_output(["osascript", "-e", script], stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except subprocess.CalledProcessError:
        return ""


def _has_gui_display() -> bool:
    # Checks if a graphical desktop session exists (needed for Tk/zenity on Linux).
    system = platform.system()
    if system == "Windows":
        return True
    if system == "Darwin":
        return True
    # Linux / other Unix: need an X/Wayland session for Tk file dialogs.
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _pick_image_tk() -> str:
    # Cross-platform Tkinter file dialog; works best on Windows/Linux with a display.
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    return (path or "").strip()


def _pick_image_zenity() -> str:
    # Linux fallback: zenity graphical file picker if the command is installed.
    exe = shutil.which("zenity")
    if not exe:
        return ""
    try:
        out = subprocess.check_output(
            [
                exe,
                "--file-selection",
                "--title=Select an image",
                "--file-filter=*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp",
                "--file-filter=*.*",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return (out or "").strip()
    except Exception:
        return ""


def _pick_image_kdialog() -> str:
    # Linux fallback: KDE kdialog file picker if available.
    exe = shutil.which("kdialog")
    if not exe:
        return ""
    try:
        # kdialog returns a quoted path; strip quotes/newlines.
        out = subprocess.check_output(
            [
                exe,
                "--getopenfilename",
                ".",
                "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp) | All files (*.*)",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        p = (out or "").strip().strip('"')
        return p
    except Exception:
        return ""


def _pick_image_interactive() -> str:
    """
    Cross-platform image selection:
    - macOS: AppleScript Finder picker (avoids Tk instability on some Python builds)
    - Windows/Linux (GUI session): Tk file dialog (fallback: zenity/kdialog on Linux)
    - otherwise: empty string (caller should fall back to manual path entry)
    """
    # Route to the right picker for this OS, then fall back to "" so caller can ask for a typed path.
    system = platform.system()
    if system == "Darwin":
        use_tk = os.environ.get("MACOS_IMAGE_PICKER", "osascript").strip().lower() in ("tk", "tkinter")
        if use_tk:
            return _pick_image_tk()
        return _pick_image_macos()

    if _has_gui_display():
        try:
            p = _pick_image_tk()
            if p:
                return p
        except Exception:
            pass

        p = _pick_image_zenity()
        if p:
            return p

        p = _pick_image_kdialog()
        if p:
            return p

    return ""


def _is_prediction_dict(obj) -> bool:
    # True if this dict has the fields we expect for a Roboflow-style bounding box.
    if not isinstance(obj, dict):
        return False
    required = {"x", "y", "width", "height"}
    if not required.issubset(set(obj.keys())):
        return False
    # confidence/class may be missing depending on workflow output
    return True


def _extract_predictions_from_workflow_result(result: Any) -> List[Dict[str, Any]]:
    """
    Workflows return nested JSON. We try to find a list of dicts that look like
    Roboflow-style predictions: {x, y, width, height, ...}.
    """
    # Walk nested workflow JSON, collect prediction lists, dedupe boxes we see more than once.
    if result is None:
        return []

    candidates: List[List[Dict[str, Any]]] = []

    def walk(node, depth: int = 0) -> None:
        # Recurse into dicts/lists to find prediction arrays; depth cap avoids runaway recursion.
        if depth > 10:
            return

        if isinstance(node, dict):
            preds = node.get("predictions")
            if isinstance(preds, list) and preds and all(_is_prediction_dict(p) for p in preds):
                candidates.append(preds)

            for v in node.values():
                walk(v, depth + 1)

        elif isinstance(node, list):
            if node and all(_is_prediction_dict(x) for x in node):
                candidates.append(node)
            for v in node:
                walk(v, depth + 1)

    walk(result)

    if not candidates:
        return []

    merged: List[Dict[str, Any]] = []
    seen = set()
    for lst in sorted(candidates, key=len, reverse=True):
        for p in lst:
            key = (
                str(p.get("class", "")),
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                float(p.get("width", 0.0)),
                float(p.get("height", 0.0)),
                float(p.get("confidence", 0.0) or 0.0),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(p)

    return merged


def _xyxy_from_rf_pred(p: Dict[str, Any]) -> Tuple[float, float, float, float]:
    # Turn Roboflow center (x,y) + width/height into corner coords (x1,y1,x2,y2).
    cx = float(p["x"])
    cy = float(p["y"])
    w = float(p["width"])
    h = float(p["height"])
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _expand_xyxy(box: Tuple[float, float, float, float], margin: float) -> Tuple[float, float, float, float]:
    # Pad or shrink the box by margin as a fraction of its width and height (for crash ROI).
    x1, y1, x2, y2 = box
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    dx = bw * margin
    dy = bh * margin
    return x1 - dx, y1 - dy, x2 + dx, y2 + dy


def _intersection_area_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    # Area of overlap between two axis-aligned rectangles; zero if they only touch or miss.
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    # Intersection-over-union: how much two boxes overlap, normalized by their combined area.
    inter = _intersection_area_xyxy(a, b)
    if inter <= 0:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    return inter / denom if denom > 0 else 0.0


def _center_in_xyxy(point: Tuple[float, float], box: Tuple[float, float, float, float]) -> bool:
    # True if the point lies inside the rectangle (used to tie a vehicle box to the crash region).
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def _is_vehicle_rf_class(name: str) -> bool:
    # Decide if a model class name should count as a vehicle (skip people and generic crash words).
    n = name.lower().strip()
    if not n:
        return False
    if "person" in n:
        return False
    # Don't treat crash-scene labels like "car accident" as a vehicle instance box.
    if any(k in n for k in ("accident", "crash", "collision", "wreck", "flip")):
        return False
    # Common plural label in datasets ("cars") should count as a vehicle box label.
    if n == "cars" or n.startswith("cars ") or n.endswith(" cars") or " cars " in n:
        return True
    return any(k in n for k in ("car", "vehicle", "truck", "bus"))


def _is_crash_rf_class(name: str) -> bool:
    # True if the label text sounds like a crash / accident / rollover type class.
    n = name.lower()
    return (
        ("crash" in n)
        or ("accident" in n)
        or ("collision" in n)
        or ("wreck" in n)
        or ("flip" in n)
        or ("hiting" in n)  # tolerate common misspelling: "hiting a tree"
        or ("hitting" in n)
    )


def _tree_hit_from_rf_class(name: str) -> bool:
    # UI hint (for roboflow ): if label mentions tree or "hitting" so we can show a tree-collision line.
    n = name.lower()
    return ("tree" in n) or ("hiting" in n) or ("hitting" in n)


def _rollover_from_rf_class(name: str) -> bool:
    # UI hint: "flip" in the class name suggests rollover wording.
    return "flip" in name.lower()


# For on-screen text for labeling images
def _overlay_font_scale(img_w: int, img_h: int) -> float:
    # Picks a font scale that looks ok on big/small images; OVERLAY_FONT_SCALE env overrides.
    env = os.environ.get("OVERLAY_FONT_SCALE", "").strip()
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    # Scale from the shorter side so very wide / short frames (common for dash/drone)
    # don't get label text that is huge relative to vertical resolution. Display-only;
    # does not affect inference.
    w = max(1, int(img_w))
    h = max(1, int(img_h))
    short = min(w, h)
    base = short / 1200.0
    cap_by_height = h / 600.0
    return float(max(0.25, min(0.78, base, cap_by_height)))


def _pil_load_sans(size_px: int) -> Any:
    """Return a Pillow ImageFont, or None if no system sans TTF is available."""
    # Load a normal sans font from well-known OS paths so labels arent bitmap-jaggy.
    try:
        from PIL import ImageFont
    except ImportError:
        return None

    size_px = int(max(10, min(96, size_px)))
    system = platform.system()
    candidates: List[str] = []
    if system == "Darwin":
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    elif system == "Windows":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        candidates = [
            os.path.join(windir, "Fonts", "arial.ttf"),
            os.path.join(windir, "Fonts", "calibri.ttf"),
            os.path.join(windir, "Fonts", "segoeui.ttf"),
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                if path.lower().endswith(".ttc"):
                    return ImageFont.truetype(path, size_px, index=0)
                return ImageFont.truetype(path, size_px)
            except OSError:
                continue
    return None


def _draw_multiline_label_pil(
    img: Any,
    lines: List[str],
    org: Tuple[int, int],
    font_scale: float,
    fg_color: Tuple[int, int, int],
    outline_color: Tuple[int, int, int],
    bg_alpha: float,
    margin: int,
    line_spacing: float,
) -> bool:
    """
    Antialiased TrueType overlay (display-only). Returns False to fall back to OpenCV Hershey text.
    """
    # Draw several lines of text with outline + tinted rectangle behind; mutates img in place.
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False

    h, w = img.shape[:2]
    short = min(w, h)
    pillow_px = int(max(14, min(64, round(font_scale * 56 + short / 900))))
    font_obj = _pil_load_sans(pillow_px)
    if font_obj is None:
        return False

    stroke_w = max(1, min(4, pillow_px // 17))
    fg = (int(fg_color[2]), int(fg_color[1]), int(fg_color[0]))
    ol = (int(outline_color[2]), int(outline_color[1]), int(outline_color[0]))

    probe = Image.new("RGB", (8, 8))
    dr0 = ImageDraw.Draw(probe)
    sizes: List[Tuple[str, int, int, Tuple[int, int, int, int]]] = []
    for ln in lines:
        bb = dr0.textbbox((0, 0), ln, font=font_obj, stroke_width=stroke_w)
        l, t, r, b = bb
        lw, lh = r - l, b - t
        sizes.append((ln, lw, lh, bb))
    text_w = max(s[1] for s in sizes)
    line_hs = [s[3][3] - s[3][1] for s in sizes]
    max_line_h = max(line_hs)
    extra_leading = max(2, pillow_px // 10)
    gap = int(round(max_line_h * (line_spacing - 1.0))) + extra_leading
    total_text_h = sum(line_hs) + gap * (len(lines) - 1)

    x0, y_top = int(org[0]), int(org[1])
    box_w = text_w + margin * 2
    y1 = y_top
    y2 = y_top + margin + total_text_h + margin
    x1 = x0 + box_w

    x0 = int(max(8, min(x0, w - box_w - 8)))
    x1 = x0 + box_w

    if y1 < 8:
        sh = 8 - y1
        y1 += sh
        y2 += sh
    if y2 > h - 8:
        sh = y2 - (h - 8)
        y1 -= sh
        y2 -= sh

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y1), (x1, y2), (15, 15, 15), thickness=-1)
    cv2.addWeighted(overlay, bg_alpha, img, 1.0 - bg_alpha, 0, dst=img)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    drw = ImageDraw.Draw(pil_img)
    y = float(y1 + margin)
    for ln, lw, lh, bb in sizes:
        tx = float(x0 + margin + (text_w - lw))
        ty = y - float(bb[1])
        drw.text((tx, ty), ln, font=font_obj, fill=fg, stroke_width=stroke_w, stroke_fill=ol)
        y += float(lh + gap)

    img[:, :, :] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return True


def _draw_multiline_label(
    img,
    text: str,
    org: Tuple[int, int],
    *,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: Optional[float] = None,
    fg_color: Tuple[int, int, int] = (245, 245, 245),
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    bg_alpha: Optional[float] = None,
    margin: int = 10,
    line_spacing: float = 1.15,
) -> None:
    # Draw a multi-line status message: try PIL for nice fonts, else OpenCV putText.
    h, w = img.shape[:2]
    if font_scale is None:
        font_scale = _overlay_font_scale(w, h)

    lines = [ln for ln in (text or "").splitlines() if ln.strip() != ""]
    if not lines:
        return

    if bg_alpha is None:
        env_a = os.environ.get("OVERLAY_BG_ALPHA", "").strip()
        if env_a:
            try:
                bg_alpha = float(env_a)
            except ValueError:
                bg_alpha = 0.55
        else:
            bg_alpha = 0.55
    bg_alpha = float(max(0.0, min(0.85, bg_alpha)))

    if not _env_bool("OVERLAY_USE_CV2_TEXT", False):
        try:
            if _draw_multiline_label_pil(
                img, lines, org, font_scale, fg_color, outline_color, bg_alpha, margin, line_spacing
            ):
                return
        except Exception:
            pass

    short_side = min(w, h)
    thickness = max(1, int(round(font_scale * 2.2)))
    if short_side < 800:
        thickness = min(thickness, 2)
    outline_thickness = thickness + (1 if short_side < 900 else 2)

    metrics = []
    for ln in lines:
        (tw, th), bl = cv2.getTextSize(ln, font, font_scale, thickness)
        metrics.append((ln, int(tw), int(th), int(bl)))

    text_w = max(m[1] for m in metrics)
    max_th = max(m[2] for m in metrics)
    max_bl = max(m[3] for m in metrics)
    extra_leading = max(2, int(round(font_scale * 6)))
    step_y = int(round((max_th + max_bl) * line_spacing + extra_leading))

    x0, y_top = org
    box_w = text_w + margin * 2

    first_baseline_y = y_top + margin + max_th
    last_baseline_y = first_baseline_y + (len(lines) - 1) * step_y

    x0 = int(max(8, min(x0, w - box_w - 8)))

    y1 = int(first_baseline_y - max_th - margin)
    y2 = int(last_baseline_y + max_bl + margin)
    x1 = x0 + box_w

    if y1 < 8:
        shift = 8 - y1
        y1 += shift
        y2 += shift
        first_baseline_y += shift
        last_baseline_y += shift
    if y2 > h - 8:
        shift = y2 - (h - 8)
        y1 -= shift
        y2 -= shift
        first_baseline_y -= shift
        last_baseline_y -= shift

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y1), (x1, y2), (15, 15, 15), thickness=-1)
    cv2.addWeighted(overlay, bg_alpha, img, 1.0 - bg_alpha, 0, dst=img)

    cur_base_y = first_baseline_y
    for ln, tw, th, bl in metrics:
        tx = int(x0 + margin + (text_w - tw))
        by = int(cur_base_y)
        cv2.putText(img, ln, (tx, by), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
        cv2.putText(img, ln, (tx, by), font, font_scale, fg_color, thickness, cv2.LINE_AA)
        cur_base_y += step_y


# App class: loads models, runs analyze_path(), handles the gui loop 


class CrashAnalysisApp:
    def __init__(
        self,
        api_key: str,
        model_id: Optional[str],
        workspace_name: Optional[str],
        workflow_id: Optional[str],
    ):
        # Wire up Roboflow HTTP client and remember model vs workflow mode; YOLO loads later.
        # YOLO is optional (often poor on wrecked / aerial imagery when using COCO pretrained weights).
        self._yolo_model = None
        self.rf_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        self.model_id = model_id
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id

    def _get_yolo(self):
        # First call loads Ultralytics YOLO from disk (weights path in env) and later calls reuse same model.
        if self._yolo_model is None:
            # Ultralytics can be chatty on stderr even with verbose=False.
            for name in ("ultralytics", "ultralytics.nn", "torch"):
                logging.getLogger(name).setLevel(logging.WARNING)
            weights = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt").strip() or "yolov8n.pt"
            self._yolo_model = YOLO(weights)
        return self._yolo_model

    def check_overlap(self, boxA, boxB):
        # Quick AABB overlap test: true if intersection area of two xyxy boxes is positive.
        xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        return interArea > 0

    def _yolo_vehicle_detections(self, frame) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        # Run YOLO on one BGR image; return list of (class name, score, xyxy box) for vehicle-like COCO classes.
        conf = _env_float("YOLO_CONF", 0.25)
        iou = _env_float("YOLO_IOU", 0.7)
        imgsz = _env_int("YOLO_IMGSZ", 960)
        include_moto = _env_bool("YOLO_INCLUDE_MOTORCYCLE", False)

        res = self._get_yolo()(frame, verbose=False, conf=conf, iou=iou, imgsz=imgsz)[0]
        names = res.names
        out: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
        for b in res.boxes:
            label = names[int(b.cls)]
            if label not in ["car", "truck", "bus"] + (["motorcycle"] if include_moto else []):
                continue
            score = float(b.conf)
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            out.append((label, score, (x1, y1, x2, y2)))
        return out

    def _yolo_obstacle_detections(self, frame) -> List[Tuple[str, float, Any]]:
        """Best-effort COCO 'obstacle-ish' classes for overlap hints (optional)."""
        # Same YOLO pass but only keep a couple COCO classes we use as weak obstacle hints in detailed UI.
        conf = _env_float("YOLO_CONF", 0.25)
        iou = _env_float("YOLO_IOU", 0.7)
        imgsz = _env_int("YOLO_IMGSZ", 960)

        res = self._get_yolo()(frame, verbose=False, conf=conf, iou=iou, imgsz=imgsz)[0]
        names = res.names
        out: List[Tuple[str, float, Any]] = []
        for b in res.boxes:
            label = names[int(b.cls)]
            if label not in ["potted plant", "stop sign"]:
                continue
            score = float(b.conf)
            coords = b.xyxy[0].cpu().numpy()
            out.append((label, score, coords))
        return out

    def _draw_yolo_vehicle_boxes(self, img, dets: List[Tuple[str, float, Tuple[float, float, float, float]]]) -> None:
        # Overlay green rectangles + labels for each YOLO vehicle detection (fallback visualization).
        ih, iw = img.shape[:2]
        fs = _overlay_font_scale(iw, ih)
        th = max(1, int(round(fs * 2)))
        if min(ih, iw) < 800:
            th = min(th, 2)
        for label, score, (x1, y1, x2, y2) in dets:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
            tag = f"{label} {score:.2f}"
            cv2.putText(
                img,
                tag,
                (int(x1), int(max(0, y1 - 6))),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs,
                (0, 200, 0),
                th,
                cv2.LINE_AA,
            )

    def _best_crash_prediction(self, predictions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # From all boxes, keep ones that look like "crash" classes and pick highest confidence.
        crash_candidates = [p for p in predictions if _is_crash_rf_class(str(p.get("class", "")))]
        if not crash_candidates:
            return None
        return max(crash_candidates, key=lambda d: float(d.get("confidence", 0) or 0.0))

    def _roboflow_predict(self, file_path: str) -> Tuple[List[Dict[str, Any]], Any]:
        """
        Returns (predictions_list, raw_payload) where predictions_list is best-effort normalized.
        """
        # Call Roboflow: either a hosted workflow (nested JSON) or a single model infer on disk path.
        use_workflow = bool(self.workspace_name and self.workflow_id)
        if use_workflow:
            raw = self.rf_client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={"image": file_path},
                use_cache=True,
            )
            preds = _extract_predictions_from_workflow_result(raw)
            return preds, raw

        if not self.model_id:
            raise ValueError("Missing ROBOFLOW_MODEL_ID (or set ROBOFLOW_WORKSPACE_NAME + ROBOFLOW_WORKFLOW_ID).")

        raw = self.rf_client.infer(file_path, model_id=self.model_id)
        preds = raw.get("predictions", []) if isinstance(raw, dict) else []
        return preds, raw

    def analyze_path(self, file_path: str) -> str:
        """
        Returns:
            "quit" if the user asked to exit from the OpenCV window (q / Esc)
            "continue" otherwise
        """
        # End-to-end for one image path: load pixels, call Roboflow (and maybe YOLO), draw result, wait for window key.
        print(f"Analyzing: {file_path}")

        # Load Image
        frame = cv2.imread(file_path)
        if frame is None:
            print(f"Failed to read image: {file_path}", file=sys.stderr)
            return "continue"
        display_frame = frame.copy()

        ui_mode = os.environ.get("CRASH_UI_MODE", "simple").strip().lower()
        if ui_mode not in ("simple", "detailed"):
            ui_mode = "simple"

        yolo_fallback = _env_bool("YOLO_FALLBACK", True)
        roboflow_fail_fallback = _env_bool("ROBOFLOW_FAIL_YOLO_FALLBACK", True)
        crash_min_conf = _env_float("CRASH_MIN_CONFIDENCE", 0.35)
        crash_margin = _env_float("CRASH_BOX_MARGIN", 0.15)
        vehicle_count_mode = os.environ.get("VEHICLE_COUNT_MODE", "roboflow").strip().lower()
        if vehicle_count_mode not in ("roboflow", "yolo", "both"):
            vehicle_count_mode = "roboflow"
        use_yolo_for_overlap = _env_bool("USE_YOLO", False)

        predictions: List[Dict[str, Any]] = []
        rf_result: Any = None

        # 2. Roboflow Accident Verification
        # Uses the api (or workflow) for predictions 
        try:
            predictions, rf_result = self._roboflow_predict(file_path)
        except Exception as e:
            print(f"API Error: Failed to connect to Roboflow: {e}", file=sys.stderr)
            predictions = []
            rf_result = None
            if not (yolo_fallback and roboflow_fail_fallback):
                return "continue"

        debug_rf = os.environ.get("DEBUG_ROBOFLOW", "0").strip().lower() in ("1", "true", "yes", "y", "on")
        if debug_rf:
            classes = [str(p.get("class", "")).strip() for p in predictions]
            print(f"Roboflow predictions: {len(predictions)}")
            # compact histogram
            hist: Dict[str, int] = {}
            for c in classes:
                hist[c] = hist.get(c, 0) + 1
            print("Class histogram:", dict(sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))))

        if not predictions and debug_rf:
            print("Roboflow returned no prediction-like objects. Raw response (for debugging):")
            print(rf_result)

        crash_pred = self._best_crash_prediction(predictions) if predictions else None
        crash_conf = float(crash_pred.get("confidence", 0) or 0.0) if crash_pred else 0.0
        crash_ok = bool(crash_pred) and crash_conf >= crash_min_conf

        # if we think theres a crash, count vehicles in the roi and draw red box
        if crash_ok:
            p_crash = crash_pred
            assert p_crash is not None

            crash_xyxy = _xyxy_from_rf_pred(p_crash)
            crash_roi = _expand_xyxy(crash_xyxy, crash_margin)

            # Vehicle counting inside crash ROI (Roboflow detections)
            rf_vehicle_preds: List[Dict[str, Any]] = []
            for pr in predictions:
                cls = str(pr.get("class", "")).strip()
                if not cls or not _is_vehicle_rf_class(cls):
                    continue
                if pr is p_crash:
                    continue
                vbox = _xyxy_from_rf_pred(pr)
                center = ((vbox[0] + vbox[2]) / 2.0, (vbox[1] + vbox[3]) / 2.0)
                if _center_in_xyxy(center, crash_roi) or _iou_xyxy(vbox, crash_roi) >= 0.05:
                    rf_vehicle_preds.append(pr)

            involved_cars: List[Any] = []
            involved_trees: List[Dict[str, Any]] = []
            if use_yolo_for_overlap:
                yolo_dets = self._yolo_vehicle_detections(frame)
                involved_cars = [d[2] for d in yolo_dets if self.check_overlap(list(crash_roi), d[2])]
                yolo_obs = self._yolo_obstacle_detections(frame)
                involved_trees = [
                    {"label": lab, "coords": coords}
                    for lab, _s, coords in yolo_obs
                    if self.check_overlap(list(crash_roi), coords)
                ]

            # Reporting Result
            rf_class = str(p_crash.get("class", "")).strip()

            if vehicle_count_mode == "yolo":
                vehicle_count = len(involved_cars)
                count_src = "YOLO"
            elif vehicle_count_mode == "both":
                vehicle_count = max(len(rf_vehicle_preds), len(involved_cars))
                count_src = "max(Roboflow,YOLO)"
            else:
                vehicle_count = len(rf_vehicle_preds)
                count_src = "Roboflow"

            if ui_mode == "simple":
                line1 = "CRASH DETECTED"
                line2 = ""
                if _tree_hit_from_rf_class(rf_class):
                    line2 = "Potential tree collision"
                elif _rollover_from_rf_class(rf_class):
                    line2 = "Potential rollover"

                # Only show counts if we actually have per-vehicle boxes (otherwise it's misleading).
                if rf_vehicle_preds or (use_yolo_for_overlap and involved_cars):
                    noun = "vehicle" if vehicle_count == 1 else "vehicles"
                    line2 = f"{vehicle_count} {noun} ({count_src})" if not line2 else f"{line2}\n{vehicle_count} {noun} ({count_src})"

                result_msg = line1 if not line2 else f"{line1}\n{line2}"
            else:
                if vehicle_count == 0 and vehicle_count_mode in ("roboflow", "both") and not rf_vehicle_preds:
                    result_msg = f"CRASH DETECTED ({count_src}): no per-vehicle boxes returned"
                else:
                    noun = "Vehicle" if vehicle_count == 1 else "Vehicles"
                    result_msg = f"CRASH DETECTED: {vehicle_count} {noun} ({count_src})"

                rf_class_lc = rf_class.lower()
                if ("tree" in rf_class_lc) or ("hiting" in rf_class_lc) or ("hitting" in rf_class_lc):
                    result_msg += " vs tree"
                elif involved_trees:
                    result_msg += f" vs {involved_trees[0]['label']}"
                elif rf_class:
                    result_msg += f" ({rf_class})"

            # Draw Results for User (REQ-6.2)
            x1, y1, x2, y2 = crash_roi
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            for pr in rf_vehicle_preds:
                vx1, vy1, vx2, vy2 = _xyxy_from_rf_pred(pr)
                cv2.rectangle(
                    display_frame,
                    (int(vx1), int(vy1)),
                    (int(vx2), int(vy2)),
                    (255, 165, 0),
                    2,
                )
            _draw_multiline_label(display_frame, result_msg, (20, 40))
            print(result_msg)
        else:
            # no strong crash from roboflow -> optional yolo pass just to show cars
            # No confident crash detection from Roboflow -> drone-friendly YOLO vehicle pass
            if crash_pred and crash_conf < crash_min_conf and debug_rf:
                print(f"No crash accepted (best crash conf={crash_conf:.3f} < {crash_min_conf:.3f}).")

            if yolo_fallback:
                yolo_dets = self._yolo_vehicle_detections(frame)
                self._draw_yolo_vehicle_boxes(display_frame, yolo_dets)

                n = len(yolo_dets)
                if ui_mode == "simple":
                    line1 = "NO CRASH DETECTED"
                    line2 = f"{n} vehicle{'s' if n != 1 else ''}" if n else "No vehicles detected"
                    _draw_multiline_label(display_frame, f"{line1}\n{line2}", (20, 40))
                else:
                    reason = "Roboflow returned no confident crash prediction."
                    if not predictions:
                        reason = "Roboflow returned no predictions."
                    noun = "vehicle" if n == 1 else "vehicles"
                    msg = f"{reason}\n{n} {noun} detected" if n else f"{reason}\nNo vehicles detected"
                    _draw_multiline_label(display_frame, msg, (20, 40))
            else:
                print("No collision detected in this image.")
                if ui_mode == "simple":
                    _draw_multiline_label(display_frame, "NO CRASH DETECTED", (20, 40))
                else:
                    _draw_multiline_label(display_frame, "No collision detected in this image.", (20, 40))

        cv2.imshow("Detection Result", display_frame)
        # waitKey(1) lets the window paint; long waitKey(0) only sees keys when this window has focus.
        cv2.waitKey(1)

        key_prompt = os.environ.get("CRASH_KEY_PROMPT", "terminal").strip().lower()
        if key_prompt not in ("terminal", "window"):
            key_prompt = "terminal"

        if key_prompt == "window":
            print("In the image window: press Q (or Esc) to quit, or any other key for another image.")
            key = cv2.waitKey(0)
            key8 = key & 0xFF
            cv2.destroyAllWindows()
            if key8 in (27, ord("q"), ord("Q")):
                print("Quit requested.")
                return "quit"
            return "continue"

        print(
            "Image window should be visible. In this terminal: press Enter for another image, "
            "or type q (or quit) then Enter to exit."
        )
        line = input().strip().lower()
        cv2.destroyAllWindows()
        if line in ("q", "quit", "exit"):
            print("Quit requested.")
            return "quit"
        return "continue"

    def run_interactive(self, loop: bool) -> None:
        # CLI loop: pick or type an image path, analyze it, repeat until user exits or --once mode.
        while True:
            system = platform.system()
            if system == "Darwin":
                if loop:
                    print("A Finder window will open to pick an image. (Cancel Finder to stop selecting files.)")
                else:
                    print("A Finder window will open to pick ONE image. Cancel Finder to exit.")
            elif _has_gui_display():
                print("A file picker window will open to select an image.")
            else:
                print("No GUI display detected; you'll be prompted to type a file path in this terminal.")

            path = _pick_image_interactive().strip()
            if not path:
                path = input("Image path (or blank to exit): ").strip()

            if not path:
                print("Exiting.")
                return

            action = self.analyze_path(path)
            if action == "quit":
                return

            if not loop:
                return


# Run from terminal: python analysis.py [--image path] [--once] ---
if __name__ == "__main__":
    # Parse CLI flags, load .env + API key, build app, then either analyze one path or interactive mode.
    parser = argparse.ArgumentParser(description="Car crash analysis (YOLO + Roboflow)")
    parser.add_argument("--image", help="Path to an image to analyze (skips file picker)")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Pick/analyze only one image in interactive mode (default is multi-image until you quit from the prompt after each result).",
    )
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    API_KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not API_KEY:
        print("Missing ROBOFLOW_API_KEY environment variable.", file=sys.stderr)
        raise SystemExit(2)

    # Model inference mode:
    #   export ROBOFLOW_MODEL_ID="your-project/3"
    #
    # Workflow mode (Roboflow snippet):
    #   export ROBOFLOW_WORKSPACE_NAME="project-workspace-h8j0c"
    #   export ROBOFLOW_WORKFLOW_ID="detect-count-and-visualize"
    #
    # After each result: default is terminal (Enter / q+Enter). Old behavior: export CRASH_KEY_PROMPT=window
    MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "").strip() or None
    WORKSPACE_NAME = os.environ.get("ROBOFLOW_WORKSPACE_NAME", "").strip() or None
    WORKFLOW_ID = os.environ.get("ROBOFLOW_WORKFLOW_ID", "").strip() or None

    app = CrashAnalysisApp(API_KEY, MODEL_ID, WORKSPACE_NAME, WORKFLOW_ID)
    if args.image:
        app.analyze_path(args.image)
    else:
        app.run_interactive(loop=not args.once)