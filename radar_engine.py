# radar_engine.py
# Core simulation + radar equation received power + guardrail clutter + effective object power

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import numpy as np


def rz(yaw_rad: float) -> np.ndarray:
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


@dataclass
class RadarConfig:
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 2.0], dtype=float))
    yaw: float = 0.0                     # radians
    fov_az: float = deg2rad(120.0)       # radians
    r_min: float = 3.0                   # close blind zone (set non-trivial so "very close range disappear" is visible)
    r_max: float = 300.0

    sigma_r: float = 0.25                # meters
    sigma_az: float = deg2rad(0.2)       # radians
    sigma_vr: float = 0.15               # m/s

    def world_to_sensor(self, p_w: np.ndarray) -> np.ndarray:
        p = p_w - self.pos
        return rz(-self.yaw) @ p

    def vel_world_to_sensor(self, v_w: np.ndarray) -> np.ndarray:
        return rz(-self.yaw) @ v_w

    def sensor_xy_to_world(self, x_s: float, y_s: float) -> np.ndarray:
        p_s = np.array([x_s, y_s, 0.0], dtype=float)
        p_w = self.pos + (rz(self.yaw) @ p_s)
        p_w[2] = 0.0
        return p_w

    def gate(self, p_s: np.ndarray) -> bool:
        x, y = float(p_s[0]), float(p_s[1])
        if x <= 0.0:
            return False
        r = float(np.linalg.norm(p_s))
        if not (self.r_min <= r <= self.r_max):
            return False
        az = math.atan2(y, x)
        if abs(az) > self.fov_az / 2.0:
            return False
        return True


@dataclass
class Participant:
    pid: int
    label: str
    pos: np.ndarray
    yaw: float
    vel: np.ndarray
    L: float
    W: float
    H: float
    sigma_m2: float               # total RCS (sigma, m^2) in this POC

    direction: str
    lane_y: float
    detection_range: Tuple[int, int] = (10, 30)  # (min, max) detections for realistic point cloud
    loop: bool = True
    active: bool = True
    desired_speed_mps: float = 12.0   # target cruise speed magnitude
    a_max: float = 2.0                # max accel (m/s^2)
    b_max: float = 4.0                # max decel (m/s^2)

    def corners_world(self) -> List[np.ndarray]:
        L, W = self.L, self.W
        corners_local = [
            np.array([+L/2, +W/2, 0.0]),
            np.array([+L/2, -W/2, 0.0]),
            np.array([-L/2, +W/2, 0.0]),
            np.array([-L/2, -W/2, 0.0]),
        ]
        R = rz(self.yaw)
        return [self.pos + (R @ c) for c in corners_local]

    def sample_surface_points(self, rng: np.random.Generator, radar_pos: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Generate realistic point cloud samples on the object's visible surfaces.
        Returns list of (world_point, point_sigma) tuples.
        
        Sampling strategy:
        - Sample points on visible faces (front/back, sides, top)
        - Weight sampling towards radar-facing surfaces
        - Add some height variation for realism
        - Higher RCS points near corners, edges, and reflective surfaces
        """
        L, W, H = self.L, self.W, self.H
        
        # Determine number of detections (distance-based scaling)
        dist_to_radar = float(np.linalg.norm(self.pos[:2] - radar_pos[:2]))
        
        # Closer objects get more detections (inverse square-ish relationship)
        # At 10m: full count; at 100m: ~1/4 of max count
        distance_factor = min(1.0, max(0.25, 10.0 / (dist_to_radar + 1.0)))
        
        min_det, max_det = self.detection_range
        n_detections = int(rng.integers(
            max(1, int(min_det * distance_factor)), 
            max(2, int(max_det * distance_factor)) + 1
        ))
        
        # Calculate which faces are visible to radar
        R = rz(self.yaw)
        radar_dir_local = rz(-self.yaw) @ (radar_pos - self.pos)
        radar_dir_local = radar_dir_local / (np.linalg.norm(radar_dir_local) + 1e-9)
        
        # Generate sample points
        points = []
        total_sigma = self.sigma_m2
        
        for _ in range(n_detections):
            # Choose which surface to sample from based on radar visibility
            # Higher probability for radar-facing surfaces
            
            # Random selection with bias towards visible faces
            face_choice = rng.random()
            
            if face_choice < 0.4:
                # Front or back face (X faces)
                if radar_dir_local[0] > 0:
                    x_local = L/2 * rng.uniform(0.7, 1.0)  # front face
                else:
                    x_local = -L/2 * rng.uniform(0.7, 1.0)  # back face
                y_local = rng.uniform(-W/2, W/2)
                z_local = rng.uniform(0.1, H * 0.9)
                
            elif face_choice < 0.75:
                # Side faces (Y faces)
                x_local = rng.uniform(-L/2, L/2)
                if radar_dir_local[1] > 0:
                    y_local = W/2 * rng.uniform(0.7, 1.0)  # right side
                else:
                    y_local = -W/2 * rng.uniform(0.7, 1.0)  # left side
                z_local = rng.uniform(0.1, H * 0.9)
                
            else:
                # Top face or corners (strong reflectors)
                x_local = rng.uniform(-L/2, L/2)
                y_local = rng.uniform(-W/2, W/2)
                z_local = H * rng.uniform(0.5, 1.0)
            
            # Add small random jitter for realism
            x_local += rng.normal(0, L * 0.02)
            y_local += rng.normal(0, W * 0.02)
            z_local = max(0.05, min(H, z_local + rng.normal(0, H * 0.02)))
            
            # Transform to world coordinates
            point_local = np.array([x_local, y_local, z_local], dtype=float)
            point_world = self.pos + (R @ point_local)
            point_world[2] = 0.0  # Project to ground for radar detection
            
            # Assign RCS to this point (distribute total sigma with variation)
            # Corners and edges have higher RCS
            corner_factor = 1.0
            if abs(abs(x_local) - L/2) < L*0.1 and abs(abs(y_local) - W/2) < W*0.1:
                corner_factor = 2.0  # Corner reflector effect
            
            base_sigma = total_sigma / n_detections
            point_sigma = base_sigma * corner_factor * rng.uniform(0.5, 1.5)
            
            points.append((point_world, point_sigma))
        
        return points

    def step(self, dt: float) -> None:
        self.pos = self.pos + self.vel * dt


@dataclass
class Detection:
    det_id: int
    participant_id: int
    label: str
    corner_id: int
    timestamp_s: float

    r: float
    az: float
    vr: float

    x_s: float
    y_s: float

    x_w: float
    y_w: float

    vx_w: float
    vy_w: float

    dx: float
    dy: float
    dvx: float
    dvy: float

    rcs_dbm: float              # received "RCS level" at radar (dBm) per this POC


@dataclass
class WorldConfig:
    x_min: float = 0.0
    x_max: float = 100.0
    lanes: int = 2
    lane_width: float = 3.5

    def lane_centers(self) -> List[float]:
        n = int(self.lanes)
        w = float(self.lane_width)
        start = -(n/2.0 - 0.5) * w
        return [start + i*w for i in range(n)]

    def road_edges(self) -> Tuple[float, float]:
        road_w = self.lane_width * self.lanes
        return -road_w/2.0, +road_w/2.0


class SimEngine:
    """
    Radar received level uses simplified radar equation:
      Pr(dBm) = K + 10log10(sigma_corner) - 40log10(R)
    Corner powers are combined per object in linear mW to compute Pr_eff_dbm.

    K is calibrated so that typical Pr_eff falls into:
      Truck: 25-32 dBm, Car: 15-25 dBm (for common mid-range distances).
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.radar = RadarConfig()
        self.world = WorldConfig()

        self.dt = 0.1
        self.t = 0.0

        self.participants: List[Participant] = []
        self.next_pid = 1
        self.next_det_id = 1

        self.random_mode = False
        self.random_max_in_scene = 2

        # --- target sigma defaults (m^2) ---
        # These are "total sigma" (distributed across detection points).
        # detection_range: [min_detections, max_detections] - realistic radar point cloud counts
        # based on typical automotive radar characteristics:
        # - Car: 15-40 detections (multiple reflectors: bumpers, wheels, license plate, etc.)
        # - Truck: 30-80 detections (large metal surface, multiple strong reflectors)
        # - Two-wheeler: 5-12 detections (engine block, wheels, handlebars)
        # - Bicycle: 2-6 detections (minimal metal, mostly frame and wheels)
        # - Pedestrian: 1-4 detections (low RCS, mostly torso reflection)
        self.defaults = {
            "car":       {"L": 4.5, "W": 1.9, "H": 1.6, "sigma_m2": 10.0, "detection_range": (15, 40)},
            "truck":     {"L": 10.0, "W": 2.6, "H": 3.5, "sigma_m2": 30.0, "detection_range": (30, 80)},
            "twowheeler": {"L": 2.2, "W": 0.8, "H": 1.5, "sigma_m2": 3.0, "detection_range": (5, 12)},   # motorcycle/scooter
            "bicycle":   {"L": 1.8, "W": 0.6, "H": 1.7, "sigma_m2": 1.0, "detection_range": (2, 6)},   # bicycle with rider
            "pedestrian": {"L": 0.5, "W": 0.5, "H": 1.75, "sigma_m2": 0.5, "detection_range": (1, 4)}, # walking person
        }

        # --- radar equation calibration ---
        # Choose K so that at R=50m:
        #   Truck (sigma_total=30): Pr_eff ≈ 28 dBm (within 25-32)
        # Corner sigma = sigma_total/4 and combining 4 corners adds ~+6 dB,
        # so using sigma_total is consistent with K≈81.2.
        self.pr_K_db = 81.2
        self.pr_shadow_sigma_db = 1.2   # slow-ish amplitude variation per detection

        # --- clutter/guardrail ---
        self.clutter_enabled = True
        self.clutter_spacing_m = 1.5
        self.clutter_offset_outside_road = 0.8
        self.clutter_jitter_xy = 0.15
        self.clutter_drop_prob = 0.08
        self.clutter_sigma_mean = 2.0     # low sigma
        self.clutter_sigma_std = 1.0

        self._clutter_anchors = []
        self._rebuild_clutter_anchors()
    def _lane_index(self, lane_y: float, tol: float = 1e-3) -> int:
        centers = self.world.lane_centers()
        diffs = [abs(lane_y - c) for c in centers]
        i = int(np.argmin(diffs))
        if diffs[i] > max(tol, 0.25 * self.world.lane_width):
            # lane_y is not close to any center, still snap to nearest
            return i
        return i

    def _lane_center(self, lane_index: int) -> float:
        return float(self.world.lane_centers()[int(lane_index)])

    def _lane_participants(self, lane_index: int) -> list[Participant]:
        y = self._lane_center(lane_index)
        # snap-by-center to be robust
        return [p for p in self.participants if abs(float(p.lane_y) - y) < 0.25 * self.world.lane_width]

    def _spawn_clearance_ok(self, lane_index: int, direction: str, L_new: float) -> bool:
        """
        Check if spawn boundary is clear for same-direction traffic.
        incoming spawns at x_max, outgoing at x_min.
        """
        lane_ps = self._lane_participants(lane_index)
        same = [p for p in lane_ps if p.direction == direction]
        if not same:
            return True

        # Minimum gap at spawn boundary (simple + safe)
        spawn_gap = 12.0 if L_new <= 5.0 else 20.0  # car vs truck-ish

        if direction == "incoming":
            x_spawn = float(self.world.x_max)
            # nearest vehicle near spawn is the one with largest x
            x_near = max(float(p.pos[0]) for p in same)
            L_near = float(max(p.L for p in same))
            # distance between bumpers at spawn edge
            gap = (x_spawn - L_new/2.0) - (x_near + L_near/2.0)
            return gap >= spawn_gap
        else:
            x_spawn = float(self.world.x_min)
            # nearest vehicle near spawn is the one with smallest x
            x_near = min(float(p.pos[0]) for p in same)
            L_near = float(max(p.L for p in same))
            gap = (x_near - L_near/2.0) - (x_spawn + L_new/2.0)
            return gap >= spawn_gap

    def _lane_has_opposite(self, lane_index: int, direction: str) -> bool:
        lane_ps = self._lane_participants(lane_index)
        opp = "incoming" if direction == "outgoing" else "outgoing"
        return any(p.direction == opp for p in lane_ps)

    def _pick_valid_lane(self, direction: str, L_new: float, preferred_lane_y: float | None = None) -> int | None:
        """
        Choose a lane that:
          - has no opposite-direction traffic
          - has spawn clearance for same-direction traffic
        If preferred is provided, try it first.
        """
        lane_indices = list(range(len(self.world.lane_centers())))

        if preferred_lane_y is not None:
            i_pref = self._lane_index(float(preferred_lane_y))
            lane_indices.remove(i_pref)
            lane_indices = [i_pref] + lane_indices

        for li in lane_indices:
            if self._lane_has_opposite(li, direction):
                continue
            if not self._spawn_clearance_ok(li, direction, L_new):
                continue
            return li
        return None

    def _apply_lane_speed_control(self, dt: float) -> None:
        """
        Prevent collisions for same-direction vehicles in the same lane.
        Simple time-headway controller:
          desired_gap = s0 + T*v
        """
        s0 = 4.0     # standstill gap (m)
        T = 1.2      # time headway (s)
        margin = 1.0

        centers = self.world.lane_centers()
        for li in range(len(centers)):
            lane_ps = self._lane_participants(li)

            for direction in ("outgoing", "incoming"):
                group = [p for p in lane_ps if p.direction == direction]
                if len(group) <= 1:
                    continue

                travel_sign = +1.0 if direction == "outgoing" else -1.0

                # sort by progress along travel direction
                # outgoing: x increasing => progress=x
                # incoming: x decreasing => progress=-x
                group.sort(key=lambda p: travel_sign * float(p.pos[0]))

                # apply control from front to back (leader ahead has higher progress)
                # group[0] is rearmost; group[-1] is frontmost in this ordering
                for idx in range(len(group) - 1):
                    follower = group[idx]
                    leader = group[idx + 1]

                    xf = float(follower.pos[0])
                    xl = float(leader.pos[0])

                    # bumpers along travel direction
                    follower_front = xf + travel_sign * (float(follower.L) / 2.0)
                    leader_rear = xl - travel_sign * (float(leader.L) / 2.0)

                    gap = travel_sign * (leader_rear - follower_front)  # positive means clear

                    v_f = abs(float(follower.vel[0]))
                    v_l = abs(float(leader.vel[0]))
                    v_des = float(max(0.0, follower.desired_speed_mps))

                    desired_gap = s0 + T * v_f

                    if gap < desired_gap:
                        # slow down strongly, and never faster than leader
                        v_new = max(0.0, v_f - follower.b_max * dt)
                        v_new = min(v_new, v_l)
                    elif gap > desired_gap + margin:
                        # accelerate toward desired speed
                        v_new = min(v_des, v_f + follower.a_max * dt)
                    else:
                        v_new = v_f  # hold

                    follower.vel[0] = travel_sign * v_new

    def _rebuild_clutter_anchors(self):
        self._clutter_anchors.clear()
        x0, x1 = self.world.x_min, self.world.x_max
        yL, yR = self.world.road_edges()
        yL -= self.clutter_offset_outside_road
        yR += self.clutter_offset_outside_road

        xs = np.arange(x0, x1 + 1e-6, self.clutter_spacing_m)
        for x in xs:
            self._clutter_anchors.append(np.array([float(x), float(yL), 0.0], dtype=float))
            self._clutter_anchors.append(np.array([float(x), float(yR), 0.0], dtype=float))

    def reset_time(self):
        self.t = 0.0
        self.next_det_id = 1

    def clear_participants(self):
        self.participants.clear()

    def set_world(self, x_min: float, x_max: float, lanes: int, lane_width: float):
        self.world.x_min = float(x_min)
        self.world.x_max = float(x_max)
        self.world.lanes = int(lanes)
        self.world.lane_width = float(lane_width)
        self._rebuild_clutter_anchors()

    def add_participant(self, label: str, direction: str, speed_mps: float, lane_y: float, loop: bool = True) -> Participant:
        d = self.defaults[label]
        L, W, H, sigma = d["L"], d["W"], d["H"], d["sigma_m2"]
        detection_range = d.get("detection_range", (10, 30))

        # pick a valid lane (no opposite direction, spawn clearance)
        li = self._pick_valid_lane(direction=direction, L_new=float(L), preferred_lane_y=float(lane_y))
        if li is None:
            raise ValueError("No valid lane available (opposite-direction present or lane too congested near spawn).")

        lane_y_use = self._lane_center(li)

        x0, vx, yaw = self._spawn_from_direction(direction, abs(speed_mps))

        p = Participant(
            pid=self.next_pid,
            label=label,
            pos=np.array([x0, lane_y_use, 0.0], dtype=float),
            yaw=yaw,
            vel=np.array([vx, 0.0, 0.0], dtype=float),
            L=float(L), W=float(W), H=float(H),
            sigma_m2=float(sigma),
            direction=direction,
            lane_y=float(lane_y_use),
            detection_range=tuple(detection_range),
            loop=bool(loop),
            active=True,
            desired_speed_mps=float(abs(speed_mps)),
        )
        self.next_pid += 1
        self.participants.append(p)
        return p


    def _spawn_from_direction(self, direction: str, speed_mps: float) -> Tuple[float, float, float]:
        x_min, x_max = self.world.x_min, self.world.x_max
        if direction == "incoming":
            return x_max, -speed_mps, math.pi
        else:
            return x_min, +speed_mps, 0.0

    def remove_participant(self, pid: int) -> None:
        self.participants = [p for p in self.participants if p.pid != pid]

    def get_participant(self, pid: int) -> Optional[Participant]:
        for p in self.participants:
            if p.pid == pid:
                return p
        return None

    def step(self) -> Dict:
        self.t = round(self.t + self.dt, 10)

        if self.random_mode:
            self._random_spawn_logic()

        # collision avoidance / car-following BEFORE integration
        self._apply_lane_speed_control(self.dt)

        for p in list(self.participants):
            p.step(self.dt)
            self._respawn_or_remove_if_outside(p)


        return self._simulate_detections()

    def _respawn_or_remove_if_outside(self, p: Participant) -> None:
        x_min, x_max = self.world.x_min, self.world.x_max
        buf = 5.0
        if p.direction == "incoming":
            if p.pos[0] < x_min - buf:
                if p.loop:
                    x0, vx, yaw = self._spawn_from_direction("incoming", abs(p.vel[0]))
                    p.pos[0] = x0
                    p.pos[1] = p.lane_y
                    p.vel[0] = vx
                    p.yaw = yaw
                else:
                    self.remove_participant(p.pid)
        else:
            if p.pos[0] > x_max + buf:
                if p.loop:
                    x0, vx, yaw = self._spawn_from_direction("outgoing", abs(p.vel[0]))
                    p.pos[0] = x0
                    p.pos[1] = p.lane_y
                    p.vel[0] = vx
                    p.yaw = yaw
                else:
                    self.remove_participant(p.pid)

    def _random_spawn_logic(self):
        if len(self.participants) >= self.random_max_in_scene:
            return
        if float(self.rng.random()) < 0.15:
            # Random label selection with weighted probabilities
            rand_val = float(self.rng.random())
            if rand_val < 0.30:
                label = "car"
                speed = float(self.rng.uniform(8.0, 18.0))
            elif rand_val < 0.50:
                label = "truck"
                speed = float(self.rng.uniform(6.0, 14.0))
            elif rand_val < 0.70:
                label = "twowheeler"
                speed = float(self.rng.uniform(6.0, 16.0))
            elif rand_val < 0.85:
                label = "bicycle"
                speed = float(self.rng.uniform(3.0, 8.0))
            else:
                label = "pedestrian"
                speed = float(self.rng.uniform(1.0, 2.0))
            direction = "incoming" if float(self.rng.random()) < 0.5 else "outgoing"

            d = self.defaults[label]
            L_new = float(d["L"])

            # pick a valid lane (no opposite direction + clearance)
            li = self._pick_valid_lane(direction=direction, L_new=L_new, preferred_lane_y=None)
            if li is None:
                return  # no lane available this tick, skip spawn

            lane_y = self._lane_center(li)
            self.add_participant(label, direction, speed, lane_y, loop=False)


    # --- radar equation helpers ---
    def _pr_dbm(self, sigma_m2: float, R_m: float) -> float:
        R_m = float(max(0.5, R_m))
        sigma_m2 = float(max(1e-6, sigma_m2))
        pr = self.pr_K_db + 10.0 * math.log10(sigma_m2) - 40.0 * math.log10(R_m)
        pr += float(self.rng.normal(0.0, self.pr_shadow_sigma_db))
        return float(pr)

    @staticmethod
    def _dbm_to_mw(dbm: float) -> float:
        return 10.0 ** (dbm / 10.0)

    @staticmethod
    def _mw_to_dbm(mw: float) -> float:
        mw = max(1e-12, float(mw))
        return 10.0 * math.log10(mw)

    def _make_detection_from_world_point(
        self,
        participant_id: int,
        label: str,
        corner_id: int,
        cw: np.ndarray,
        vel_world: np.ndarray,
        sigma_corner_m2: float
    ) -> Optional[Detection]:
        ps = self.radar.world_to_sensor(cw)
        if not self.radar.gate(ps):
            return None

        r_true = float(np.linalg.norm(ps))
        az_true = float(math.atan2(ps[1], ps[0]))
        v_rel_s = self.radar.vel_world_to_sensor(vel_world)
        u = ps / (r_true + 1e-9)
        vr_true = float(np.dot(v_rel_s, u))

        # noisy measurement
        r_m = r_true + float(self.rng.normal(0.0, self.radar.sigma_r))
        r_m = float(max(0.5, r_m))
        az_m = az_true + float(self.rng.normal(0.0, self.radar.sigma_az))
        vr_m = vr_true + float(self.rng.normal(0.0, self.radar.sigma_vr))

        x_s = r_m * math.cos(az_m)
        y_s = r_m * math.sin(az_m)

        pw = self.radar.sensor_xy_to_world(x_s, y_s)
        x_w, y_w = float(pw[0]), float(pw[1])

        # approx velocity from vr along LOS
        los_world = rz(self.radar.yaw) @ np.array([math.cos(az_m), math.sin(az_m), 0.0], dtype=float)
        vx_est = float(vr_m * los_world[0])
        vy_est = float(vr_m * los_world[1])

        x_true, y_true = float(cw[0]), float(cw[1])
        dx = x_w - x_true
        dy = y_w - y_true

        vx_true, vy_true = float(vel_world[0]), float(vel_world[1])
        dvx = vx_est - vx_true
        dvy = vy_est - vy_true

        # received power for this corner (dBm)
        rcs_dbm = self._pr_dbm(sigma_corner_m2, r_m)

        det = Detection(
            det_id=self.next_det_id,
            participant_id=participant_id,
            label=label,
            corner_id=corner_id,
            timestamp_s=float(self.t),
            r=float(r_m), az=float(az_m), vr=float(vr_m),
            x_s=float(x_s), y_s=float(y_s),
            x_w=float(x_w), y_w=float(y_w),
            vx_w=float(vx_est), vy_w=float(vy_est),
            dx=float(dx), dy=float(dy), dvx=float(dvx), dvy=float(dvy),
            rcs_dbm=float(rcs_dbm),
        )
        self.next_det_id += 1
        return det

    def _simulate_detections(self) -> Dict:
        detections: List[Detection] = []
        objects_out: List[Dict] = []
        nearest: Optional[Detection] = None
        
        # participants - using realistic point cloud sampling
        for p in self.participants:
            # Sample realistic number of points on visible surfaces
            surface_points = p.sample_surface_points(self.rng, self.radar.pos)

            point_meas_world_xy = []
            pr_mw_sum = 0.0

            for point_id, (point_world, point_sigma) in enumerate(surface_points):
                det = self._make_detection_from_world_point(
                    participant_id=p.pid,
                    label=p.label,
                    corner_id=point_id,  # using corner_id for point_id
                    cw=point_world,
                    vel_world=p.vel,
                    sigma_corner_m2=point_sigma
                )
                if det is None:
                    continue

                detections.append(det)
                point_meas_world_xy.append((det.x_w, det.y_w))

                # combine per object effective power
                pr_mw_sum += self._dbm_to_mw(det.rcs_dbm)

                if nearest is None or det.r < nearest.r:
                    nearest = det

            pr_eff_dbm = None
            if pr_mw_sum > 0:
                pr_eff_dbm = float(self._mw_to_dbm(pr_mw_sum))

            bbox_world = None
            if len(point_meas_world_xy) > 0:
                xs = [pt[0] for pt in point_meas_world_xy]
                ys = [pt[1] for pt in point_meas_world_xy]
                bbox_world = {
                    "xmin": float(min(xs)),
                    "xmax": float(max(xs)),
                    "ymin": float(min(ys)),
                    "ymax": float(max(ys)),
                    "z": 0.0
                }

            objects_out.append({
                "participant_id": p.pid,
                "label": p.label,
                "pos": p.pos.copy(),
                "yaw": float(p.yaw),
                "dims": (float(p.L), float(p.W), float(p.H)),
                "sigma_m2": float(p.sigma_m2),
                "pr_eff_dbm": pr_eff_dbm,           # <-- effective combined power
                "loop": bool(p.loop),
                "direction": p.direction,
                "lane_y": float(p.lane_y),
                "bbox_world": bbox_world,
                "vel": p.vel.copy(),
                "speed_mps": float(np.linalg.norm(p.vel[:2])),
                "detection_count": len(point_meas_world_xy),  # track actual detections

            })

        # clutter / guardrail points
        if self.clutter_enabled:
            vel0 = np.array([0.0, 0.0, 0.0], dtype=float)
            for anchor in self._clutter_anchors:
                if float(self.rng.random()) < self.clutter_drop_prob:
                    continue
                jitter = self.rng.normal(0.0, self.clutter_jitter_xy, size=(2,))
                cw = anchor.copy()
                cw[0] += float(jitter[0])
                cw[1] += float(jitter[1])

                sigma = float(max(1e-3, self.rng.normal(self.clutter_sigma_mean, self.clutter_sigma_std)))
                det = self._make_detection_from_world_point(
                    participant_id=0,
                    label="clutter",
                    corner_id=-1,
                    cw=cw,
                    vel_world=vel0,
                    sigma_corner_m2=sigma
                )
                if det is None:
                    continue
                detections.append(det)
                if nearest is None or det.r < nearest.r:
                    nearest = det

        return {
            "t_s": float(self.t),
            "radar": {
                "pos": self.radar.pos.copy(),
                "yaw": float(self.radar.yaw),
                "fov_az": float(self.radar.fov_az),
                "r_min": float(self.radar.r_min),
                "r_max": float(self.radar.r_max),
            },
            "world": {
                "x_min": float(self.world.x_min),
                "x_max": float(self.world.x_max),
                "lanes": int(self.world.lanes),
                "lane_width": float(self.world.lane_width),
                "lane_centers": self.world.lane_centers(),
            },
            "detections": detections,
            "objects": objects_out,
            "nearest": nearest,  # kept for debugging; UI no longer plots it
        }
