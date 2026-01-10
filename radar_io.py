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
