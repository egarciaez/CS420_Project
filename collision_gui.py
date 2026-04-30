import tkinter as tk
from tkinter import ttk
import os
import cv2
import datetime
import tempfile
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from drone_navigation import DroneNavigation
from analysis import (
    CrashAnalysisApp,
    _env_float,
    _env_bool,
    _expand_xyxy,
    _xyxy_from_rf_pred,
    _draw_multiline_label,
    _is_vehicle_rf_class,
    _tree_hit_from_rf_class,
    _rollover_from_rf_class
)

class CollisionAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Collision Analysis System - Emergency Dashboard") # [cite: 348]
        self.root.geometry("1000x800")
        self.root.configure(bg="#1e1e1e") # Dark background to match wireframe

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TButton", padding=5, relief="flat", background="#00a8cc", foreground="white", font=("Arial", 10, "bold"))
        style.configure("TLabel", background="#1e1e1e", foreground="white", font=("Arial", 10))
        style.configure("Header.TLabel", font=("Arial", 12, "bold"), foreground="#00a8cc")
        style.configure("Status.TLabel", font=("Arial", 10, "bold"), foreground="#ff4c4c")


        self.left_panel = ttk.Frame(self.root, padding=10)
        self.left_panel.grid(row=0, column=0, sticky="nsew") # Removed weight=1
        
        self.right_panel = ttk.Frame(self.root, padding=10)
        self.right_panel.grid(row=0, column=1, sticky="nsew") # Removed weight=3

        self.root.columnconfigure(0, weight=1) # Left panel gets 1 part of the width
        self.root.columnconfigure(1, weight=3) # Right panel gets 3 parts of the width
        self.root.rowconfigure(0, weight=1)    # Both fill the vertical space

        self._build_left_panel()
        self._build_right_panel()

        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")
        model_id = os.environ.get("ROBOFLOW_MODEL_ID", None)
        workspace = os.environ.get("ROBOFLOW_WORKSPACE_NAME", None)
        workflow = os.environ.get("ROBOFLOW_WORKFLOW_ID", None)
        
        self.log_to_console("Connecting to backend...")
        self.backend_app = CrashAnalysisApp(api_key, model_id, workspace, workflow)
        self.log_to_console("System initialized and ready.")

        self.scan_active = False
        self.cap = None
        self._tello = None
        self._tello_frame_reader = None
        self._tello_help_shown = False
        self._manual_override_active = False
        self._manual_control_window = None

        self._drone_nav = None
        self._drone_sim_running = False

    def _connect_tello(self) -> bool:
        """
        Best-effort connect to a DJI Tello / Tello EDU using djitellopy.
        Returns True on success; logs and returns False on failure.
        """
        try:
            from djitellopy import Tello  # type: ignore
        except Exception as e:
            self.log_to_console(f"Tello connection unavailable (install djitellopy): {e}")
            return False

        try:
            tello = Tello()
            tello.connect()
            try:
                batt = tello.get_battery()
                self.log_to_console(f"Tello connected (battery {batt}%). Starting video stream...")
            except Exception:
                self.log_to_console("Tello connected. Starting video stream...")
            tello.streamon()
            frame_reader = tello.get_frame_read()
            self._tello = tello
            self._tello_frame_reader = frame_reader
            return True
        except Exception as e:
            self.log_to_console(f"Error: Could not connect to Tello stream: {e}")
            try:
                if self._tello is not None:
                    self._tello.end()
            except Exception:
                pass
            self._tello = None
            self._tello_frame_reader = None
            return False

    def _disconnect_tello(self) -> None:
        tello = self._tello
        self._tello = None
        self._tello_frame_reader = None
        if tello is None:
            return
        try:
            try:
                tello.streamoff()
            except Exception:
                pass
            try:
                # Ensure we stop any RC control stream when disconnecting.
                tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass
            tello.end()
        except Exception:
            pass

    def _tello_send_rc(self, lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0) -> None:
        """Safe wrapper around send_rc_control (values in [-100,100])."""
        if self._tello is None:
            return
        try:
            self._tello.send_rc_control(int(lr), int(fb), int(ud), int(yaw))
        except Exception as e:
            self.log_to_console(f"Tello RC control failed: {e}")

    def _tello_command(self, fn_name: str, *args) -> None:
        """Call a tello method by name with basic error handling."""
        tello = self._tello
        if tello is None:
            return
        fn = getattr(tello, fn_name, None)
        if fn is None:
            self.log_to_console(f"Tello command not supported: {fn_name}")
            return
        try:
            fn(*args)
        except Exception as e:
            self.log_to_console(f"Tello command failed ({fn_name}): {e}")

    def _build_left_panel(self):

        ttk.Label(self.left_panel, text="Controls", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Button(self.left_panel, text="Start Scan", command=self.start_scan).pack(fill="x", pady=2) # [cite: 350]
        ttk.Button(self.left_panel, text="Stop Scan", command=self.stop_scan).pack(fill="x", pady=2) # [cite: 351]
        ttk.Button(self.left_panel, text="View Report", command=self.view_report).pack(fill="x", pady=(2, 15)) # [cite: 352]


        ttk.Label(self.left_panel, text="Camera Options", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Button(self.left_panel, text="Live Feed", command=self.toggle_live_feed).pack(fill="x", pady=2) # [cite: 354]
        ttk.Button(self.left_panel, text="Upload Image", command=self.upload_image).pack(fill="x", pady=2) # [cite: 355]
        ttk.Button(self.left_panel, text="Take Picture", command=self.take_picture).pack(fill="x", pady=(2, 15)) # [cite: 356]


        ttk.Label(self.left_panel, text="Status", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        self.status_label = ttk.Label(self.left_panel, text="Cars Involved: [ 0 ]", style="Status.TLabel") # [cite: 363]
        self.status_label.pack(anchor="w", pady=2)
        ttk.Button(self.left_panel, text="Run Simulation", command=self.start_drone_simulation).pack(fill="x", pady=(10, 2))
        ttk.Button(self.left_panel, text="Manual Override", command=self.manual_override).pack(fill="x", pady=(10, 15)) # [cite: 184]

    def _build_right_panel(self):

        ttk.Label(self.right_panel, text="Drone Camera Feed", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        self.camera_canvas = tk.Canvas(self.right_panel, bg="#a0a0a0", height=250) # [cite: 358]
        self.camera_canvas.pack(fill="both", expand=True, pady=(0, 15))
        self.camera_canvas.create_text(300, 125, text="Live Video / Uploaded Image Placeholder", fill="white", font=("Arial", 12))



        ttk.Label(self.right_panel, text="Safety Perimeter Simulation", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        self.sim_canvas = tk.Canvas(self.right_panel, bg="#a0a0a0", width=900, height=260, highlightthickness=0)
        self.sim_canvas.pack(fill="x", expand=False, pady=(0, 15))
        self.sim_canvas.create_text(
            450, 130,
            text="Simulation will run here when a crash is detected",
            fill="white",
            font=("Arial", 12),
        )

        ttk.Label(self.right_panel, text="System Console / Feedback", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        self.console_text = tk.Text(self.right_panel, height=8, bg="#005f73", fg="white", font=("Courier", 10), state="disabled")
        self.console_text.pack(fill="both", expand=False)
        
        # Insert initial startup text [cite: 372]
        self.log_to_console("System initializing...")

    # --- Utility Methods ---
    def log_to_console(self, message):
        """Helper method to add text to the read-only console."""
        self.console_text.config(state="normal")
        self.console_text.insert(tk.END, message + "\n")
        self.console_text.see(tk.END)
        self.console_text.config(state="disabled")

    
    def display_frame_on_canvas(self, cv2_frame):
        """Converts an OpenCV BGR frame and safely displays it on the Tkinter canvas."""
        if cv2_frame is None:
            return

        # convert OpenCV BGR to Tkinter RGB
        rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # get current canvas dimensions, with fallbacks
        canvas_width = self.camera_canvas.winfo_width()
        canvas_height = self.camera_canvas.winfo_height()
        if canvas_width <= 1: canvas_width = 600
        if canvas_height <= 1: canvas_height = 250
            
        # resize and render
        pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(pil_image) 

        self.camera_canvas.delete("all")
        x_center = canvas_width // 2
        y_center = canvas_height // 2
        self.camera_canvas.create_image(x_center, y_center, anchor="center", image=self.photo_image)

    def save_to_report(self, filename, status_message):
        """Appends scan results to the local text file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("crash_report_log.txt", "a") as file:
            file.write(f"[{timestamp}] File: {os.path.basename(filename)}\n")
            # Replaces newlines with dashes so the log file stays easy to read
            file.write(f"Result: {status_message.replace(chr(10), ' - ')}\n") 
            file.write("-" * 40 + "\n")

    def process_image_for_gui(self, file_path: str):
        frame = cv2.imread(file_path)
        if frame is None:
            return None, f"Failed to read image: {file_path}"
        
        display_frame = frame.copy()
        car_count = 0 
        crash_type_text = "" 

        try:
         predictions, _ = self.backend_app._roboflow_predict(file_path)
        except Exception as e:
            self.log_to_console(f"Roboflow unavailable: {e}. Falling back to YOLO...")
            predictions = []

        crash_pred = self.backend_app._best_crash_prediction(predictions) if predictions else None
        crash_conf = float(crash_pred.get("confidence", 0) or 0.0) if crash_pred else 0.0
        crash_min_conf = _env_float("CRASH_MIN_CONFIDENCE", 0.35)
        
        if crash_pred and crash_conf >= crash_min_conf:
            crash_roi = _expand_xyxy(_xyxy_from_rf_pred(crash_pred), _env_float("CRASH_BOX_MARGIN", 0.15))
            x1, y1, x2, y2 = crash_roi
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            
            rf_class = str(crash_pred.get("class", "")).strip()
            
            if _tree_hit_from_rf_class(rf_class):
                crash_type_text = "\nType: Potential tree collision"
            elif _rollover_from_rf_class(rf_class):
                crash_type_text = "\nType: Potential rollover"
            elif rf_class:
                crash_type_text = f"\nType: {rf_class.title()}"

            for pr in predictions:
                cls_name = str(pr.get("class", "")).strip()
                if _is_vehicle_rf_class(cls_name) and pr is not crash_pred:
                    car_count += 1
                    vx1, vy1, vx2, vy2 = _xyxy_from_rf_pred(pr)
                    cv2.rectangle(display_frame, (int(vx1), int(vy1)), (int(vx2), int(vy2)), (255, 165, 0), 2)

            # if Roboflow found 0 cars, double check with YOLO
            if car_count == 0:
                yolo_dets = self.backend_app._yolo_vehicle_detections(frame)
                involved_cars = []
                
                for det in yolo_dets:
                    label, score, coords = det
                    if self.backend_app.check_overlap(list(crash_roi), coords):
                        involved_cars.append(det)
                        car_count += 1
                        
                if involved_cars:
                    self.backend_app._draw_yolo_vehicle_boxes(display_frame, involved_cars)

            result_msg = f"CRASH DETECTED{crash_type_text}"
            
        else:
            # fallback to YOLO if no crash is found anywhere in the image
            if _env_bool("YOLO_FALLBACK", True):
                yolo_dets = self.backend_app._yolo_vehicle_detections(frame)
                self.backend_app._draw_yolo_vehicle_boxes(display_frame, yolo_dets)
                car_count = len(yolo_dets)
                result_msg = f"NO CRASH DETECTED\n{car_count} vehicles detected."
            else:
                result_msg = "NO CRASH DETECTED"

        if result_msg.startswith("CRASH DETECTED"):
            self.status_label.config(text=f"CRASH DETECTED{crash_type_text}")
            self._maybe_trigger_drone_simulation()
        else:
            self.status_label.config(text="NO CRASH DETECTED")

        _draw_multiline_label(display_frame, result_msg, (20, 40))
        
        return display_frame, result_msg

    def _maybe_trigger_drone_simulation(self):
        # Avoid retriggering on every live-feed frame while a mission is already running.
        if self._drone_sim_running:
            return
        self.start_drone_simulation()

    def start_drone_simulation(self):
        # Stops any previous mission loop, clears the simulation canvas,
        # then runs the navigation animation to a crash marker.
        if self._drone_nav is not None:
            self._drone_nav.is_finished = True

        self.sim_canvas.delete("all")
        self._drone_nav = DroneNavigation(self.sim_canvas)
        nav = self._drone_nav

        raw_crash_location = nav.random_road_location()
        crash_location = nav.find_valid_crash_location(*raw_crash_location)

        self.sim_canvas.create_oval(
            crash_location[0] - 10,
            crash_location[1] - 10,
            crash_location[0] + 10,
            crash_location[1] + 10,
            fill="#CC2B2B",
            outline="#7A1414",
            width=2,
            tags="crash_marker",
        )
        self.sim_canvas.create_text(
            crash_location[0],
            crash_location[1] + 18,
            text="CRASH",
            fill="#7A1414",
            font=("Arial", 9, "bold"),
            tags="crash_marker",
        )

        self._drone_sim_running = True
        self.log_to_console("Navigating to collision site...")

        def _on_arrival(location):
            self.log_to_console("Arrived at collision site. Deploying cones...")
            self._drone_sim_running = False
            self._draw_cones_around(location)

        nav.start(crash_location, _on_arrival)

    def _draw_cones_around(self, location):
        cx, cy = location
        cone_points = [
            (cx - 55, cy - 25),
            (cx - 10, cy - 55),
            (cx + 40, cy - 30),
            (cx + 55, cy + 25),
            (cx + 0, cy + 55),
            (cx - 40, cy + 30),
        ]
        for x, y in cone_points:
            self.sim_canvas.create_oval(
                x - 12, y - 12, x + 12, y + 12,
                fill="#C28B2E",
                outline="#6E4B16",
                width=2,
                tags="cone",
            )
            self.sim_canvas.create_text(
                x, y,
                text="Cone",
                fill="white",
                font=("Arial", 8, "bold"),
                tags="cone",
            )



    # --- Placeholder Functions for Future Backend Connections ---

    def start_scan(self):
        if self.scan_active:
            self.log_to_console("Scan is already running.")
            return
            
        use_tello = _env_bool("USE_TELLO", False)
        self.log_to_console("Scan Started: Connecting to camera...")
        self.scan_active = True

        if use_tello:
            if not self._tello_help_shown:
                self._tello_help_shown = True
                messagebox.showinfo(
                    "Connect to Tello Wi‑Fi",
                    "To use the Tello EDU camera stream:\n\n"
                    "1) Power on the Tello.\n"
                    "2) On your PC, connect to the Tello's Wi‑Fi (SSID usually starts with 'TELLO-').\n"
                    "3) Then click Start Scan again if needed.\n\n"
                    "If it can’t connect, the app will fall back to your webcam.",
                )
            if not self._connect_tello():
                self.log_to_console("Falling back to webcam...")
                use_tello = False

        if not use_tello:
            # open the default webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.log_to_console("Error: Could not open camera feed.")
                self.scan_active = False
                return
            
        # continuous loop
        self.update_scan()

    def stop_scan(self):
        self.log_to_console("Scan Stopped.")
        self.scan_active = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self._disconnect_tello()
            
        self.camera_canvas.delete("all")
        self.camera_canvas.create_text(300, 125, text="Live Feed Stopped", fill="white", font=("Arial", 12))

    def update_scan(self):
        # stop the loop if the user clicked "Stop Scan" or the camera isn't initialized
        if not self.scan_active:
            return

        frame = None
        ret = False
        if self._tello_frame_reader is not None:
            # djitellopy returns frames as numpy arrays; they are usable directly by OpenCV.
            frame = self._tello_frame_reader.frame
            ret = frame is not None
        elif self.cap is not None:
            ret, frame = self.cap.read()
        
        if ret:
            if self._manual_override_active:
                # Pilot mode: keep showing the live stream, but pause ML inference.
                self.display_frame_on_canvas(frame)
            else:
                # Save the frame to a true temp location (NOT inside OneDrive),
                # so live scanning doesn't constantly churn synced files.
                temp_filename = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        temp_filename = tmp.name
                    cv2.imwrite(temp_filename, frame)

                    # process the temporary image just like an uploaded image
                    processed_frame, result_message = self.process_image_for_gui(temp_filename)
                finally:
                    if temp_filename and os.path.exists(temp_filename):
                        try:
                            os.remove(temp_filename)
                        except OSError:
                            pass

                # save to the report log ONLY if a crash is detected to avoid spamming the log
                if "CRASH DETECTED" in result_message:
                    self.save_to_report("Live Feed", result_message)

                self.display_frame_on_canvas(processed_frame)

        else:
            # camera disconnects or the video file ends
            self.log_to_console("Warning: Lost connection to camera feed.")
            self.stop_scan()
            return

        # loop function every second
        # delay keeps the GUI responsive and prevents API rate-limiting
        self.root.after(1000, self.update_scan)
        
    def view_report(self):
        self.log_to_console("Opening Report View...")
        report_window = tk.Toplevel(self.root)
        report_window.title("Collision Analysis Reports")
        report_window.geometry("600x400")
        report_window.configure(bg="#1e1e1e")
        
        ttk.Label(report_window, text="System Crash Log", font=("Arial", 14, "bold"), background="#1e1e1e", foreground="#00a8cc").pack(pady=10)
        
        # scrolling text area
        text_area = tk.Text(report_window, wrap="word", bg="#2d2d2d", fg="white", font=("Courier", 10))
        text_area.pack(expand=True, fill="both", padx=10, pady=(0, 10))
        
        # read the locally saved report file
        try:
            with open("crash_report_log.txt", "r") as file:
                logs = file.read()
                if logs.strip() == "":
                    text_area.insert(tk.END, "No crashes have been logged yet.")
                else:
                    text_area.insert(tk.END, logs)
        except FileNotFoundError:
            text_area.insert(tk.END, "No report file found. Run a scan to generate the first report.")
            
        # make the text read-only initially
        text_area.config(state="disabled")

        # clear the file and the screen
        def _clear_and_refresh():
            # clear text file
            with open("crash_report_log.txt", "w") as file:
                file.write("")
            self.log_to_console("Report history has been cleared.")
            
            # u the text box, clear the screen, add placeholder, and re-lock
            text_area.config(state="normal")
            text_area.delete("1.0", tk.END)
            text_area.insert(tk.END, "No crashes have been logged yet.")
            text_area.config(state="disabled")

        clear_btn = ttk.Button(report_window, text="Clear", command=_clear_and_refresh)
        clear_btn.place(x=10, y=10)
        
    def toggle_live_feed(self):
        self.log_to_console("Toggling Live Video Feed...")
        # TODO: Connect OpenCV VideoCapture feed to the camera_canvas [cite: 180]
        
    def upload_image(self):
        if self.scan_active:
            self.stop_scan()
            self.log_to_console("Live feed paused for manual image upload.")

        file_path = filedialog.askopenfilename(
            title="Select a Crash Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return 
            
        self.log_to_console(f"Processing uploaded image: {os.path.basename(file_path)}...")
        self.root.update() 


        frame, result_message = self.process_image_for_gui(file_path)
        
        if frame is None:
            self.log_to_console(result_message)
            return


        self.save_to_report(file_path, result_message)


        self.display_frame_on_canvas(frame)
        
        self.log_to_console(f"Analysis Complete: {result_message.replace(chr(10), ' - ')}")
        
    def take_picture(self):
        self.log_to_console("Commanding Drone to Take Picture...")
        # TODO: Send command to drone hardware/simulation to capture a frame [cite: 181]
        
    def manual_override(self):
        # Pilot mode: user manually controls the drone.
        if self._manual_override_active:
            self._manual_override_active = False
            self.log_to_console("Manual Override Deactivated (automation resumed).")
            try:
                self._tello_send_rc(0, 0, 0, 0)
            except Exception:
                pass
            if self._manual_control_window is not None:
                try:
                    self._manual_control_window.destroy()
                except Exception:
                    pass
                self._manual_control_window = None
            return

        if self._tello is None:
            messagebox.showwarning(
                "Manual Override",
                "No Tello is connected.\n\n"
                "To enable manual control:\n"
                "- Connect to the Tello Wi‑Fi (SSID usually starts with 'TELLO-')\n"
                "- Set USE_TELLO=1 in your .env\n"
                "- Click Start Scan to connect\n",
            )
            self.log_to_console("Manual Override requested, but no Tello is connected.")
            return

        self._manual_override_active = True
        self.log_to_console("Manual Override Activated (pilot mode). ML inference paused.")
        self._open_manual_control_window()

    def _open_manual_control_window(self):
        if self._manual_control_window is not None:
            try:
                self._manual_control_window.lift()
                return
            except Exception:
                self._manual_control_window = None

        win = tk.Toplevel(self.root)
        win.title("Manual Drone Control (Tello)")
        win.geometry("440x380")
        win.configure(bg="#1e1e1e")
        self._manual_control_window = win

        header = ttk.Label(win, text="Pilot Mode Controls", style="Header.TLabel")
        header.pack(anchor="w", padx=10, pady=(10, 6))

        ttk.Label(
            win,
            text="Keyboard: WASD=move, QE=rotate, RF=up/down, X=stop, T=takeoff, L=land\nFlips: I=forward, K=back, J=left, O=right",
            wraplength=410,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 10))

        speed = tk.IntVar(value=40)  # RC control speed

        def _btn(text, cmd):
            b = ttk.Button(win, text=text, command=cmd)
            return b

        grid = ttk.Frame(win)
        grid.pack(padx=10, pady=(0, 10), fill="x")

        # Movement buttons (send brief RC nudge)
        def _nudge(lr=0, fb=0, ud=0, yaw=0, ms=250):
            self._tello_send_rc(lr, fb, ud, yaw)
            self.root.after(ms, lambda: self._tello_send_rc(0, 0, 0, 0))

        s = lambda: int(speed.get())

        _btn("Takeoff (T)", lambda: self._tello_command("takeoff")).pack(in_=grid, side="left", padx=4, pady=4)
        _btn("Land (L)", lambda: self._tello_command("land")).pack(in_=grid, side="left", padx=4, pady=4)
        _btn("STOP (X)", lambda: self._tello_send_rc(0, 0, 0, 0)).pack(in_=grid, side="left", padx=4, pady=4)

        move = ttk.Frame(win)
        move.pack(padx=10, pady=(0, 10))

        _btn("Up (R)", lambda: _nudge(ud=s())).grid(in_=move, row=0, column=1, padx=4, pady=4)
        _btn("Forward (W)", lambda: _nudge(fb=s())).grid(in_=move, row=1, column=1, padx=4, pady=4)
        _btn("Left (A)", lambda: _nudge(lr=-s())).grid(in_=move, row=2, column=0, padx=4, pady=4)
        _btn("Back (S)", lambda: _nudge(fb=-s())).grid(in_=move, row=2, column=1, padx=4, pady=4)
        _btn("Right (D)", lambda: _nudge(lr=s())).grid(in_=move, row=2, column=2, padx=4, pady=4)
        _btn("Down (F)", lambda: _nudge(ud=-s())).grid(in_=move, row=3, column=1, padx=4, pady=4)

        _btn("Rotate CCW (Q)", lambda: _nudge(yaw=-s())).grid(in_=move, row=1, column=0, padx=4, pady=4)
        _btn("Rotate CW (E)", lambda: _nudge(yaw=s())).grid(in_=move, row=1, column=2, padx=4, pady=4)

        ttk.Label(win, text="Speed").pack(anchor="w", padx=10)
        ttk.Scale(win, from_=10, to=100, variable=speed, orient="horizontal").pack(fill="x", padx=10, pady=(0, 10))

        flips = ttk.Frame(win)
        flips.pack(padx=10, pady=(0, 10), fill="x")
        ttk.Label(flips, text="Flips").pack(anchor="w")

        flip_row = ttk.Frame(flips)
        flip_row.pack(anchor="w", pady=(4, 0))
        _btn("Flip Forward (I)", lambda: self._tello_command("flip_forward")).pack(in_=flip_row, side="left", padx=4)
        _btn("Flip Back (K)", lambda: self._tello_command("flip_back")).pack(in_=flip_row, side="left", padx=4)
        _btn("Flip Left (J)", lambda: self._tello_command("flip_left")).pack(in_=flip_row, side="left", padx=4)
        _btn("Flip Right (O)", lambda: self._tello_command("flip_right")).pack(in_=flip_row, side="left", padx=4)

        def _on_close():
            # Leaving pilot mode on close is dangerous; deactivate it.
            self._manual_override_active = False
            try:
                self._tello_send_rc(0, 0, 0, 0)
            except Exception:
                pass
            self.log_to_console("Manual Override window closed (automation resumed).")
            try:
                win.destroy()
            except Exception:
                pass
            self._manual_control_window = None

        win.protocol("WM_DELETE_WINDOW", _on_close)

        # Keyboard bindings (active while the window is focused)
        def _bind(key, fn):
            win.bind(key, lambda _e: fn())

        _bind("<KeyPress-t>", lambda: self._tello_command("takeoff"))
        _bind("<KeyPress-l>", lambda: self._tello_command("land"))
        _bind("<KeyPress-x>", lambda: self._tello_send_rc(0, 0, 0, 0))

        _bind("<KeyPress-w>", lambda: _nudge(fb=s()))
        _bind("<KeyPress-s>", lambda: _nudge(fb=-s()))
        _bind("<KeyPress-a>", lambda: _nudge(lr=-s()))
        _bind("<KeyPress-d>", lambda: _nudge(lr=s()))
        _bind("<KeyPress-r>", lambda: _nudge(ud=s()))
        _bind("<KeyPress-f>", lambda: _nudge(ud=-s()))
        _bind("<KeyPress-q>", lambda: _nudge(yaw=-s()))
        _bind("<KeyPress-e>", lambda: _nudge(yaw=s()))

        _bind("<KeyPress-i>", lambda: self._tello_command("flip_forward"))
        _bind("<KeyPress-k>", lambda: self._tello_command("flip_back"))
        _bind("<KeyPress-j>", lambda: self._tello_command("flip_left"))
        _bind("<KeyPress-o>", lambda: self._tello_command("flip_right"))

        win.focus_set()


# --- Application Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = CollisionAnalysisGUI(root)
    root.mainloop()