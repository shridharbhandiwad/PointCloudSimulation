# radar_io.py
# UDP streaming + CSV/TXT logging + Excel export

from __future__ import annotations
import os
import csv
import json
import socket
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

try:
    import openpyxl
except Exception:
    openpyxl = None


FIELDS = [
    "time_ms",
    "det_id",
    "participant_id",
    "label",
    "corner_id",
    "x",
    "y",
    "vx",
    "vy",
    "rcs_dbm",
    "dx",
    "dy",
    "dvx",
    "dvy",
    "r",
    "az_deg",
    "vr",
]

# Object-level feature fields logged each cycle
OBJECT_FIELDS = [
    "time_ms",
    "participant_id",
    "label",
    "pos_x",
    "pos_y",
    "pos_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "speed_mps",
    "yaw_deg",
    "length",
    "width",
    "height",
    "sigma_m2",
    "pr_eff_dbm",
    "range_m",
    "azimuth_deg",
    "direction",
    "lane_y",
    "num_detections",
]


@dataclass
class UdpConfig:
    enabled: bool = False
    ip: str = "127.0.0.1"
    port: int = 5005
    as_json: bool = True


class UdpSender:
    def __init__(self, cfg: UdpConfig):
        self.cfg = cfg
        self.sock: Optional[socket.socket] = None

    def start(self):
        if not self.cfg.enabled:
            return
        if self.sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def stop(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None

    def send_rows(self, rows: List[Dict[str, Any]]):
        if not self.cfg.enabled:
            return
        if self.sock is None:
            self.start()
        addr = (self.cfg.ip, int(self.cfg.port))

        for row in rows:
            if self.cfg.as_json:
                payload = (json.dumps(row) + "\n").encode("utf-8")
            else:
                payload = (",".join(str(row.get(k, "")) for k in FIELDS) + "\n").encode("utf-8")
            try:
                self.sock.sendto(payload, addr)
            except Exception:
                pass


class CsvLogger:
    def __init__(self):
        self.enabled = False
        self.path_csv: Optional[str] = None
        self.fp = None
        self.writer = None

    def start(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.path_csv = filepath
        self.fp = open(filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fp, fieldnames=FIELDS)
        self.writer.writeheader()
        self.fp.flush()
        self.enabled = True

    def stop(self):
        self.enabled = False
        if self.fp:
            try:
                self.fp.flush()
                self.fp.close()
            except Exception:
                pass
        self.fp = None
        self.writer = None

    def write_rows(self, rows: List[Dict[str, Any]]):
        if not self.enabled or self.writer is None:
            return
        for r in rows:
            self.writer.writerow({k: r.get(k, "") for k in FIELDS})
        self.fp.flush()


class ObjectFeatureLogger:
    """
    Logs object-level features (position, velocity, RCS, etc.) for each object each radar cycle.
    """
    def __init__(self):
        self.enabled = False
        self.path_csv: Optional[str] = None
        self.fp = None
        self.writer = None

    def start(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.path_csv = filepath
        self.fp = open(filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fp, fieldnames=OBJECT_FIELDS)
        self.writer.writeheader()
        self.fp.flush()
        self.enabled = True

    def stop(self):
        self.enabled = False
        if self.fp:
            try:
                self.fp.flush()
                self.fp.close()
            except Exception:
                pass
        self.fp = None
        self.writer = None

    def write_rows(self, rows: List[Dict[str, Any]]):
        if not self.enabled or self.writer is None:
            return
        for r in rows:
            self.writer.writerow({k: r.get(k, "") for k in OBJECT_FIELDS})
        self.fp.flush()

    def export_xlsx(self, xlsx_path: str) -> bool:
        if self.path_csv is None:
            return False
        if openpyxl is None:
            return False

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "object_features"
        ws.append(OBJECT_FIELDS)

        import csv as _csv
        with open(self.path_csv, "r", newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                ws.append([row.get(k, "") for k in OBJECT_FIELDS])

        os.makedirs(os.path.dirname(xlsx_path) or ".", exist_ok=True)
        wb.save(xlsx_path)
        return True

    def export_xlsx(self, xlsx_path: str) -> bool:
        if self.path_csv is None:
            return False
        if openpyxl is None:
            return False

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "radar_output"
        ws.append(FIELDS)

        import csv as _csv
        with open(self.path_csv, "r", newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                ws.append([row.get(k, "") for k in FIELDS])

        os.makedirs(os.path.dirname(xlsx_path) or ".", exist_ok=True)
        wb.save(xlsx_path)
        return True


def detections_to_rows(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    dets = frame.get("detections", [])
    t_ms = int(round(frame["t_s"] * 1000.0))

    import math
    for d in dets:
        rows.append({
            "time_ms": t_ms,
            "det_id": d.det_id,
            "participant_id": d.participant_id,
            "label": d.label,
            "corner_id": d.corner_id,
            "x": round(float(d.x_w), 4),
            "y": round(float(d.y_w), 4),
            "vx": round(float(d.vx_w), 4),
            "vy": round(float(d.vy_w), 4),
            "rcs_dbm": round(float(d.rcs_dbm), 3),
            "dx": round(float(d.dx), 4),
            "dy": round(float(d.dy), 4),
            "dvx": round(float(d.dvx), 4),
            "dvy": round(float(d.dvy), 4),
            "r": round(float(d.r), 4),
            "az_deg": round(float(d.az) * 180.0 / math.pi, 4),
            "vr": round(float(d.vr), 4),
        })
    return rows


def objects_to_feature_rows(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert frame objects to feature rows for object-level logging.
    Logs features for each object each radar cycle.
    """
    rows = []
    objects = frame.get("objects", [])
    dets = frame.get("detections", [])
    radar = frame.get("radar", {})
    t_ms = int(round(frame["t_s"] * 1000.0))

    import math

    # Get radar position for range/azimuth calculation
    radar_pos = radar.get("pos", [0.0, 0.0, 0.0])
    radar_x = float(radar_pos[0])
    radar_y = float(radar_pos[1])

    # Count detections per participant
    det_counts = {}
    for d in dets:
        pid = d.participant_id
        if pid not in det_counts:
            det_counts[pid] = 0
        det_counts[pid] += 1

    for o in objects:
        pid = o["participant_id"]
        pos = o["pos"]
        vel = o.get("vel", [0.0, 0.0, 0.0])
        dims = o.get("dims", (0.0, 0.0, 0.0))

        # Calculate range and azimuth from radar to object
        dx = float(pos[0]) - radar_x
        dy = float(pos[1]) - radar_y
        range_m = math.hypot(dx, dy)
        azimuth_deg = math.degrees(math.atan2(dy, dx))

        # Get effective power (pr_eff_dbm) if available
        pr_eff = o.get("pr_eff_dbm", None)
        pr_eff_str = round(float(pr_eff), 3) if pr_eff is not None else ""

        rows.append({
            "time_ms": t_ms,
            "participant_id": pid,
            "label": o["label"],
            "pos_x": round(float(pos[0]), 4),
            "pos_y": round(float(pos[1]), 4),
            "pos_z": round(float(pos[2]), 4),
            "vel_x": round(float(vel[0]), 4),
            "vel_y": round(float(vel[1]), 4),
            "vel_z": round(float(vel[2]), 4),
            "speed_mps": round(float(o.get("speed_mps", 0.0)), 4),
            "yaw_deg": round(math.degrees(float(o.get("yaw", 0.0))), 4),
            "length": round(float(dims[0]), 4),
            "width": round(float(dims[1]), 4),
            "height": round(float(dims[2]), 4),
            "sigma_m2": round(float(o.get("sigma_m2", 0.0)), 4),
            "pr_eff_dbm": pr_eff_str,
            "range_m": round(range_m, 4),
            "azimuth_deg": round(azimuth_deg, 4),
            "direction": o.get("direction", ""),
            "lane_y": round(float(o.get("lane_y", 0.0)), 4),
            "num_detections": det_counts.get(pid, 0),
        })

    return rows
