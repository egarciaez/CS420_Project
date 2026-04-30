import random


class DroneNavigation:
    """Drone search-and-approach controller for a static city map."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.canvas_width = int(float(self.canvas.cget("width")))
        self.canvas_height = int(float(self.canvas.cget("height")))

        self.drone_tag = "drone"
        self.scan_waypoints = []
        self.current_scan_waypoint_index = 0
        self.navigation_state = "idle"
        self.on_arrival = None
        self.is_finished = False
        self.detour_waypoint = None

        # Buildings/skyscrapers that block line of sight and motion planning.
        # Trees are intentionally visual-only (the drone flies over them).
        self.blocking_structures = []
        self.horizontal_road_top_y = self.canvas_height * 0.42
        self.horizontal_road_bottom_y = self.canvas_height * 0.62
        self.vertical_road_width = 95
        self.intersection_center_x = self.canvas_width * 0.5
        self.vertical_road_left_x = self.intersection_center_x - self.vertical_road_width / 2
        self.vertical_road_right_x = self.intersection_center_x + self.vertical_road_width / 2

        # Draw map and spawn drone sprite.
        self.create_background()
        self.create_obstacles()
        self.create_drone()

    def create_background(self):
        self.canvas.create_rectangle(
            0, 0, self.canvas_width, self.canvas_height, fill="#8CCF7E", outline=""
        )
        # Horizontal road
        self.canvas.create_rectangle(
            0,
            self.horizontal_road_top_y,
            self.canvas_width,
            self.horizontal_road_bottom_y,
            fill="#5B5F62",
            outline="",
        )
        # Vertical road to create a 4-way intersection
        v_left = self.vertical_road_left_x
        v_right = self.vertical_road_right_x
        self.canvas.create_rectangle(v_left, 0, v_right, self.canvas_height, fill="#5B5F62", outline="")

        stripe_y = (self.horizontal_road_top_y + self.horizontal_road_bottom_y) / 2
        stripe_width = 30
        stripe_gap = 20
        # Lane markers are visual details only; navigation ignores them.
        x = 10
        while x < self.canvas_width:
            self.canvas.create_rectangle(x, stripe_y - 2, x + stripe_width, stripe_y + 2, fill="#E6D35C", outline="")
            x += stripe_width + stripe_gap

        stripe_x = (v_left + v_right) / 2
        y = 10
        while y < self.canvas_height:
            self.canvas.create_rectangle(stripe_x - 2, y, stripe_x + 2, y + stripe_width, fill="#E6D35C", outline="")
            y += stripe_width + stripe_gap

        # Add stop signs around intersection corners
        sign_positions = [
            (v_left - 20, self.horizontal_road_top_y - 20),
            (v_right + 20, self.horizontal_road_top_y - 20),
            (v_left - 20, self.horizontal_road_bottom_y + 20),
            (v_right + 20, self.horizontal_road_bottom_y + 20),
        ]
        for sx, sy in sign_positions:
            self.canvas.create_line(sx, sy + 5, sx, sy + 24, fill="#8A8A8A", width=2)
            self.canvas.create_polygon(
                sx - 7,
                sy - 7,
                sx + 7,
                sy - 7,
                sx + 10,
                sy,
                sx + 7,
                sy + 7,
                sx - 7,
                sy + 7,
                sx - 10,
                sy,
                fill="#C62828",
                outline="#8D1D1D",
            )
            self.canvas.create_text(sx, sy, text="STOP", fill="white", font=("Arial", 5, "bold"))

    def create_drone(self):
        # Start from top-left corner
        start_x = 35
        start_y = 35
        body_w = 24
        body_h = 14
        arm = 12
        rotor_r = 5
        x1 = start_x - body_w / 2
        y1 = start_y - body_h / 2
        x2 = start_x + body_w / 2
        y2 = start_y + body_h / 2

        self.canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            fill="#2F7FBF",
            outline="#1D4B75",
            width=2,
            tags=self.drone_tag,
        )
        self.canvas.create_line(x1 - arm, start_y, x2 + arm, start_y, fill="#1D4B75", width=2, tags=self.drone_tag)
        self.canvas.create_line(start_x, y1 - arm, start_x, y2 + arm, fill="#1D4B75", width=2, tags=self.drone_tag)

        for rx, ry in [
            (x1 - arm, start_y),
            (x2 + arm, start_y),
            (start_x, y1 - arm),
            (start_x, y2 + arm),
        ]:
            self.canvas.create_oval(
                rx - rotor_r,
                ry - rotor_r,
                rx + rotor_r,
                ry + rotor_r,
                fill="#BFC7CF",
                outline="#7D8893",
                tags=self.drone_tag,
            )

        self.canvas.create_oval(start_x - 3, start_y - 3, start_x + 3, start_y + 3, fill="#FFFFFF", outline="", tags=self.drone_tag)

    def create_obstacles(self):
        # Fixed obstacle layout (same every run).
        tree_positions = [(120, 140), (270, 50), (650, 485), (790, 510)]
        building_specs = [
            (250, 185, 44, 118, True),
            (330, 185, 52, 62, False),
            (150, 480, 54, 64, False),
            (300, 510, 58, 68, False),
            (620, 130, 46, 122, True),
            (770, 155, 44, 118, True),
        ]

        # Trees: larger trunk + canopy (visual only, not blocking).
        for tree_center_x, tree_center_y in tree_positions:
            trunk_width = 12
            trunk_height = 50
            canopy_radius = 17
            trunk_top_y = tree_center_y - 2
            trunk_bottom_y = trunk_top_y + trunk_height
            self.canvas.create_rectangle(
                tree_center_x - trunk_width / 2,
                trunk_top_y,
                tree_center_x + trunk_width / 2,
                trunk_bottom_y,
                fill="#7C4A27",
                outline="#5E351C",
            )
            self.canvas.create_oval(
                tree_center_x - canopy_radius,
                trunk_top_y - 20,
                tree_center_x + canopy_radius,
                trunk_top_y + 14,
                fill="#2E8B57",
                outline="#206642",
            )

        # Buildings: fixed regular + skyscrapers on grass only.
        for building_center_x, building_center_y, building_width, building_height, is_skyscraper in building_specs:
            x1 = building_center_x - building_width / 2
            y1 = building_center_y - building_height / 2
            x2 = building_center_x + building_width / 2
            y2 = building_center_y + building_height / 2
            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill="#8B959F" if not is_skyscraper else "#6F7A86",
                outline="#5F6870",
                width=2,
            )
            roof_h = 10 if not is_skyscraper else 6
            self.canvas.create_rectangle(x1 - 3, y1 - roof_h, x2 + 3, y1, fill="#6E747C", outline="#4B5056")
            window_rows = 2 if not is_skyscraper else 6
            window_cols = 2
            for row in range(window_rows):
                for col in range(window_cols):
                    wx1 = x1 + 8 + col * (building_width / 2 - 2)
                    wy1 = y1 + 10 + row * (building_height / (window_rows + 1))
                    self.canvas.create_rectangle(wx1, wy1, wx1 + 8, wy1 + 10, fill="#DDE7F0", outline="#93A2AF")
            # Only these rectangles are used by collision and line-of-sight logic.
            self.blocking_structures.append((x1, y1 - roof_h, x2, y2))

    def get_drone_center(self):
        x1, y1, x2, y2 = self.canvas.bbox(self.drone_tag)
        return (x1 + x2) / 2, (y1 + y2) / 2

    def clamp_move_to_canvas(self, dx, dy):
        x1, y1, x2, y2 = self.canvas.bbox(self.drone_tag)
        margin = 2

        if x1 + dx < margin:
            dx = margin - x1
        elif x2 + dx > self.canvas_width - margin:
            dx = (self.canvas_width - margin) - x2

        if y1 + dy < margin:
            dy = margin - y1
        elif y2 + dy > self.canvas_height - margin:
            dy = (self.canvas_height - margin) - y2

        return dx, dy

    def start(self, target, callback):
        # Reset mission state before beginning a new route.
        safe_target = self.find_valid_crash_location(target[0], target[1])
        self.crash_site_x, self.crash_site_y = safe_target
        self.on_arrival = callback
        # Start in search mode and clear previous mission progress.
        self.navigation_state = "searching"
        self.is_finished = False
        self.detour_waypoint = None
        self.current_scan_waypoint_index = 0
        self.scan_waypoints = self.generate_search_path()

        self.move()

    def generate_search_path(self):
        # Zig-zag sweep helps the drone gain line-of-sight before direct approach.
        path = []
        margin = 50
        top_band = max(55, int(self.horizontal_road_top_y - 45))
        lanes = [top_band, top_band + 30, top_band + 60]
        for i, lane_y in enumerate(lanes):
            if i % 2 == 0:
                path.append((self.canvas_width - margin, lane_y))
            else:
                path.append((margin, lane_y))
        return path

    def point_inside_rect(self, point_x, point_y, rect, padding=0):
        x1, y1, x2, y2 = rect
        return x1 - padding <= point_x <= x2 + padding and y1 - padding <= point_y <= y2 + padding

    def segment_hits_rect(self, start_x, start_y, end_x, end_y, rect, padding=0):
        # Approximate intersection test: sample points along segment.
        # This is simpler than exact geometry and works well for this simulation.
        steps = 28
        for i in range(steps + 1):
            t = i / steps
            sample_x = start_x + (end_x - start_x) * t
            sample_y = start_y + (end_y - start_y) * t
            if self.point_inside_rect(sample_x, sample_y, rect, padding):
                return True
        return False

    def get_blocking_structure(self, from_x, from_y, to_x, to_y):
        # Return only the nearest blocker along the segment.
        blockers = []
        for x1, y1, x2, y2 in self.blocking_structures:
            if self.segment_hits_rect(from_x, from_y, to_x, to_y, (x1, y1, x2, y2), padding=10):
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                distance_from_start = ((center_x - from_x) ** 2 + (center_y - from_y) ** 2) ** 0.5
                blockers.append((distance_from_start, x1, y1, x2, y2))
        if not blockers:
            return None
        blockers.sort(key=lambda item: item[0])
        return blockers[0][1:]

    def has_line_of_sight_to_crash(self):
        current_x, current_y = self.get_drone_center()
        return self.get_blocking_structure(current_x, current_y, self.crash_site_x, self.crash_site_y) is None

    def point_in_structure(self, x, y, padding=8):
        for x1, y1, x2, y2 in self.blocking_structures:
            if self.point_inside_rect(x, y, (x1, y1, x2, y2), padding=padding):
                return True
        return False

    def is_on_road(self, x, y, padding=0):
        on_horizontal = self.horizontal_road_top_y + padding <= y <= self.horizontal_road_bottom_y - padding
        on_vertical = self.vertical_road_left_x + padding <= x <= self.vertical_road_right_x - padding
        return on_horizontal or on_vertical

    def random_road_location(self):
        # Random crash on road or intersection only.
        if random.random() < 0.5:
            rx = random.randint(35, self.canvas_width - 35)
            ry = random.randint(int(self.horizontal_road_top_y) + 14, int(self.horizontal_road_bottom_y) - 14)
        else:
            rx = random.randint(int(self.vertical_road_left_x) + 14, int(self.vertical_road_right_x) - 14)
            ry = random.randint(35, self.canvas_height - 35)
        return rx, ry

    def find_valid_crash_location(self, target_x, target_y):
        # Crash must be on road and outside blocking structures.
        if self.is_on_road(target_x, target_y, padding=8) and not self.point_in_structure(target_x, target_y, padding=12):
            return target_x, target_y

        # Fallback: resample until we find a safe road point.
        for _ in range(300):
            rx, ry = self.random_road_location()
            if not self.point_in_structure(rx, ry, padding=12):
                return rx, ry

        return self.random_road_location()

    def choose_detour_waypoint(self, blocker, goal_x, goal_y, current_x, current_y):
        # Score side-waypoints around the blocking structure and pick best.
        x1, y1, x2, y2 = blocker
        margin = 36
        candidate_points = [
            (x1 - margin, (y1 + y2) / 2),  # left
            (x2 + margin, (y1 + y2) / 2),  # right
            ((x1 + x2) / 2, y1 - margin),  # up
            ((x1 + x2) / 2, y2 + margin),  # down
        ]
        edge_pad = 24
        scored = []
        for candidate_x, candidate_y in candidate_points:
            candidate_x = max(edge_pad, min(self.canvas_width - edge_pad, candidate_x))
            candidate_y = max(edge_pad, min(self.canvas_height - edge_pad, candidate_y))
            if self.point_in_structure(candidate_x, candidate_y, padding=10):
                continue

            edge_clearance = min(
                candidate_x,
                self.canvas_width - candidate_x,
                candidate_y,
                self.canvas_height - candidate_y,
            )

            # Scoring:
            # +140 if current -> candidate is blocked
            # +70 if candidate -> goal is blocked
            # -small bonus for staying away from map edges
            path_blocked = self.get_blocking_structure(current_x, current_y, candidate_x, candidate_y) is not None
            to_goal_blocked = self.get_blocking_structure(candidate_x, candidate_y, goal_x, goal_y) is not None
            distance_score = ((goal_x - candidate_x) ** 2 + (goal_y - candidate_y) ** 2) ** 0.5

            score = distance_score
            if path_blocked:
                score += 140
            if to_goal_blocked:
                score += 70
            score -= edge_clearance * 0.2
            scored.append((score, candidate_x, candidate_y))

        if scored:
            scored.sort(key=lambda item: item[0])
            return scored[0][1], scored[0][2]

        # Edge-safe fallback if every side is constrained.
        fallback_x = min(max(goal_x, edge_pad), self.canvas_width - edge_pad)
        fallback_y = min(max(goal_y, edge_pad), self.canvas_height - edge_pad)
        return fallback_x, fallback_y

    def move_toward(self, target_x, target_y, speed=4.5):
        current_x, current_y = self.get_drone_center()
        goal_x, goal_y = target_x, target_y

        # If a detour is active, keeps following it before retrying direct travel.
        if self.detour_waypoint:
            detour_x, detour_y = self.detour_waypoint
            detour_dist = ((detour_x - current_x) ** 2 + (detour_y - current_y) ** 2) ** 0.5

            if detour_dist <= 16:
                self.detour_waypoint = None
            else:
                target_x, target_y = detour_x, detour_y

        if self.detour_waypoint is None:
            blocker = self.get_blocking_structure(current_x, current_y, goal_x, goal_y)
            if blocker:
                self.detour_waypoint = self.choose_detour_waypoint(blocker, goal_x, goal_y, current_x, current_y)
                target_x, target_y = self.detour_waypoint

        offset_x = target_x - current_x
        offset_y = target_y - current_y
        dist = (offset_x * offset_x + offset_y * offset_y) ** 0.5
        if dist == 0:
            return 0, 0, True
        step = min(speed, dist)
        dx = offset_x / dist * step
        dy = offset_y / dist * step
        dx, dy = self.clamp_move_to_canvas(dx, dy)
        reached = dist <= speed
        return dx, dy, reached

    def move(self):
        if self.is_finished:
            return

        # Two parts: searches (scans) first, then final approach to crash site.
        if self.navigation_state == "searching":
            # 1st: sweep waypoints until direct line-of-sight is clear.
            if self.has_line_of_sight_to_crash():
                self.navigation_state = "to_crash"
            else:
                if self.current_scan_waypoint_index < len(self.scan_waypoints):
                    waypoint_x, waypoint_y = self.scan_waypoints[self.current_scan_waypoint_index]
                    dx, dy, reached = self.move_toward(waypoint_x, waypoint_y, speed=4)
                    self.canvas.move(self.drone_tag, dx, dy)
                    if reached:
                        self.current_scan_waypoint_index += 1
                    self.canvas.after(35, self.move)
                    return

                self.navigation_state = "to_crash"

        # 2nd: approaches crash site with detours around structures.
        dx, dy, reached = self.move_toward(self.crash_site_x, self.crash_site_y, speed=5)
        self.canvas.move(self.drone_tag, dx, dy)

        if reached:
            self.is_finished = True
            if self.on_arrival:
                self.on_arrival((self.crash_site_x, self.crash_site_y))
            return

        self.canvas.after(35, self.move)


if __name__ == "__main__":
    import tkinter as tk

    root = tk.Tk()
    root.title("Drone Navigation Simulation")

    canvas = tk.Canvas(root, width=900, height=600, highlightthickness=0)
    canvas.pack()
    controls = tk.Frame(root)
    controls.pack(pady=6)

    def done(location):
        print("Reached:", location)

    nav_holder = {"nav": None}

    def run_simulation():
        previous_nav = nav_holder["nav"]
        if previous_nav is not None:
            previous_nav.is_finished = True
        canvas.delete("drone")
        canvas.delete("crash_marker")

        nav = DroneNavigation(canvas)
        nav_holder["nav"] = nav

        raw_crash_location = nav.random_road_location()
        crash_location = nav.find_valid_crash_location(*raw_crash_location)

        canvas.create_oval(
            crash_location[0] - 10,
            crash_location[1] - 10,
            crash_location[0] + 10,
            crash_location[1] + 10,
            fill="#CC2B2B",
            outline="#7A1414",
            width=2,
            tags="crash_marker",
        )
        canvas.create_text(
            crash_location[0],
            crash_location[1] + 18,
            text="CRASH",
            fill="#7A1414",
            font=("Arial", 9, "bold"),
            tags="crash_marker",
        )

        nav.start(crash_location, done)

    tk.Button(controls, text="Run Again", command=run_simulation).pack()

    run_simulation()
    root.mainloop()

