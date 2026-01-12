# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 18:16:56 2026

@author: okran
"""

# ui_components.py
# Shared UI components + helpers (colors, meshes, bird-eye) + FOV sector mesh

from __future__ import annotations
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import math


from radar_engine import rad2deg


def rcs_to_rgba(vals: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Blue->Cyan->Green->Yellow->Red heat scale.
    """
    if vals.size == 0:
        return np.zeros((0, 4), dtype=float)

    vmin = float(vmin)
    vmax = float(vmax)
    if vmax <= vmin + 1e-9:
        vmax = vmin + 1.0

    t = (vals - vmin) / (vmax - vmin)
    t = np.clip(t, 0.0, 1.0)

    anchors = np.array([
        [0.00, 0.00, 0.25, 1.00],
        [0.25, 0.00, 1.00, 1.00],
        [0.50, 0.00, 1.00, 0.00],
        [0.75, 1.00, 1.00, 0.00],
        [1.00, 1.00, 0.00, 0.00],
    ], dtype=float)

    rgb = np.zeros((len(t), 3), dtype=float)
    for i, ti in enumerate(t):
        for k in range(len(anchors) - 1):
            t0 = anchors[k, 0]
            t1 = anchors[k + 1, 0]
            if ti <= t1 or k == len(anchors) - 2:
                a = (ti - t0) / (t1 - t0 + 1e-12)
                c0 = anchors[k, 1:4]
                c1 = anchors[k + 1, 1:4]
                rgb[i] = (1 - a) * c0 + a * c1
                break

    alpha = np.ones((len(t), 1), dtype=float)
    return np.hstack([rgb, alpha])


def make_cuboid_mesh(L: float, W: float, H: float) -> gl.MeshData:
    x = L / 2.0
    y = W / 2.0
    z0 = 0.0
    z1 = H

    verts = np.array([
        [-x, -y, z0], [ x, -y, z0], [ x,  y, z0], [-x,  y, z0],
        [-x, -y, z1], [ x, -y, z1], [ x,  y, z1], [-x,  y, z1],
    ], dtype=float)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=int)

    return gl.MeshData(vertexes=verts, faces=faces)


def make_cuboid_wireframe(L: float, W: float, H: float) -> np.ndarray:
    """
    Create wireframe edges for a cuboid (for transparent object visualization).
    Returns array of line segments suitable for GLLinePlotItem.
    """
    x = L / 2.0
    y = W / 2.0
    z0 = 0.0
    z1 = H

    # 8 vertices
    v = np.array([
        [-x, -y, z0], [ x, -y, z0], [ x,  y, z0], [-x,  y, z0],
        [-x, -y, z1], [ x, -y, z1], [ x,  y, z1], [-x,  y, z1],
    ], dtype=float)

    # 12 edges of a cuboid (connected as line strip)
    # Bottom face, top face, vertical edges
    edges = np.array([
        # Bottom face
        v[0], v[1], v[1], v[2], v[2], v[3], v[3], v[0],
        # Top face
        v[4], v[5], v[5], v[6], v[6], v[7], v[7], v[4],
        # Vertical edges
        v[0], v[4], v[1], v[5], v[2], v[6], v[3], v[7],
    ], dtype=float)

    return edges


# Label to color mapping for transparent objects with wireframe
LABEL_COLORS = {
    "car": (0.7, 0.7, 0.7, 0.15),       # Light gray, very transparent
    "truck": (0.9, 0.9, 0.9, 0.15),     # White-ish, very transparent  
    "twowheeler": (1.0, 0.85, 0.4, 0.15),  # Yellow, very transparent
    "bicycle": (0.4, 1.0, 0.4, 0.15),   # Green, very transparent
    "pedestrian": (1.0, 0.4, 1.0, 0.15),  # Magenta, very transparent
}

LABEL_EDGE_COLORS = {
    "car": (0.8, 0.8, 0.8, 0.8),       # Light gray edges
    "truck": (1.0, 1.0, 1.0, 0.9),     # White edges
    "twowheeler": (1.0, 0.9, 0.5, 0.9),  # Yellow edges
    "bicycle": (0.5, 1.0, 0.5, 0.9),   # Green edges
    "pedestrian": (1.0, 0.5, 1.0, 0.9),  # Magenta edges
}


def set_item_pose(item: gl.GLGraphicsItem, pos_xyz: np.ndarray, yaw_rad: float):
    from PyQt5 import QtGui
    m = QtGui.QMatrix4x4()
    m.translate(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]))
    m.rotate(rad2deg(yaw_rad), 0, 0, 1)
    item.setTransform(m)


def bbox_polyline_world(bbox: dict):
    if bbox is None:
        return None
    xmin, xmax = bbox["xmin"], bbox["xmax"]
    ymin, ymax = bbox["ymin"], bbox["ymax"]
    z = bbox.get("z", 0.12)
    pts = np.array([
        [xmin, ymin, z],
        [xmax, ymin, z],
        [xmax, ymax, z],
        [xmin, ymax, z],
        [xmin, ymin, z],
    ], dtype=float)
    return pts


def build_fov_sector_mesh(radar, n: int = 40, z: float = 0.03):
    """
    Creates a filled ground-projected sector wedge from r_min..r_max (fan).
    Returns (verts, faces, outline_pts).
    """
    rmin = float(radar.r_min)
    rmax = float(radar.r_max)
    fov = float(radar.fov_az)

    # angles in sensor frame
    ang = np.linspace(-fov/2.0, +fov/2.0, n)

    # outer arc points (world)
    outer = []
    for a in ang:
        xs = rmax * np.cos(a)
        ys = rmax * np.sin(a)
        pw = radar.sensor_xy_to_world(xs, ys)
        outer.append([pw[0], pw[1], z])

    # inner arc points (world) (optional ring; helps show near blind zone)
    inner = []
    for a in ang[::-1]:
        xs = rmin * np.cos(a)
        ys = rmin * np.sin(a)
        pw = radar.sensor_xy_to_world(xs, ys)
        inner.append([pw[0], pw[1], z])

    poly = np.array(outer + inner, dtype=float)

    # triangulate fan via simple ear-like approach using center point (approx)
    # use radar ground point as anchor
    center = np.array([radar.pos[0], radar.pos[1], z], dtype=float)

    verts = np.vstack([center, poly])
    faces = []
    # triangles: center -> i -> i+1 (for outer arc only)
    # We'll build using the outer segment to keep it simple & stable.
    outer_count = len(outer)
    for i in range(outer_count - 1):
        faces.append([0, 1 + i, 1 + i + 1])
    faces = np.array(faces, dtype=int)

    # outline: show both rmin and rmax boundaries
    outline_outer = np.array(outer, dtype=float)
    outline_inner = np.array(inner[::-1], dtype=float)  # reverse back to same direction
    return verts, faces, (outline_outer, outline_inner)


class BirdsEyeWindow(QtWidgets.QMainWindow):
    """
    BEV convention:
    - Plot X axis = lateral (world Y)
    - Plot Y axis = forward (world X)
    - Forward increases upwards.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bird's Eye View (Reference Orientation)")
        self.resize(820, 620)

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.setBackground("k")
        self.setCentralWidget(self.plot)

        # Make it zoomable/pannable
        self.plot.setMouseEnabled(x=True, y=True)

        # Hide axes for a "sensor UI" look
        self.plot.showAxis("left", False)
        self.plot.showAxis("bottom", False)

        # Ensure no accidental axis inversion
        self.plot.getViewBox().invertX(False)
        self.plot.getViewBox().invertY(False)

        # Items
        self.road_item = pg.PlotDataItem()
        self.lanes_items = []

        self.veh_scatter = pg.ScatterPlotItem(size=14)
        self.det_scatter = pg.ScatterPlotItem(size=6)

        self.bbox_curves = {}
        self.bbox_labels = {}  # pid -> TextItem for object type + confidence

        self.plot.addItem(self.road_item)
        self.plot.addItem(self.veh_scatter)
        self.plot.addItem(self.det_scatter)

        # HUD text overlay
        self.hud = pg.TextItem(anchor=(0, 0), color=(220, 220, 220))
        self.plot.addItem(self.hud)
        self.hud.setZValue(10)

        self._first_frame = True  # used for default zoom only once

    def _set_default_view_once(self, radar, world):
        """
        Default zoom: show 0..0.5*r_max forward from radar, and full road width laterally.
        """
        if not self._first_frame:
            return
        self._first_frame = False

        rmax = float(radar["r_max"])
        xr = float(radar["pos"][0])
        yr = float(radar["pos"][1])

        # Forward (world X) range shown: radar_x .. radar_x + 0.5*rmax
        y_min = xr - 5.0
        y_max = xr + 0.5 * rmax

        # Lateral (world Y): cover road width with padding
        lanes = int(world["lanes"])
        lane_w = float(world["lane_width"])
        road_w = lanes * lane_w
        x_min = -road_w/2 - road_w*0.6
        x_max = +road_w/2 + road_w*0.6

        # NOTE: plot X is lateral (world Y); plot Y is forward (world X)
        self.plot.setXRange(x_min, x_max, padding=0.0)
        self.plot.setYRange(y_min, y_max, padding=0.0)

    def update_view(self, frame: dict, show_bbox: bool, show_dets: bool, class_results: dict = None, show_classification: bool = True):
        radar = frame["radar"]
        world = frame["world"]
        objects = frame["objects"]
        dets = frame["detections"]

        # Apply default zoom once (still zoomable afterwards)
        self._set_default_view_once(radar, world)

        # --- draw road ---
        x0, x1 = world["x_min"], world["x_max"]
        lane_centers = world["lane_centers"]
        lane_width = world["lane_width"]
        road_w = lane_width * world["lanes"]
        y0, y1 = -road_w / 2.0, +road_w / 2.0

        road = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=float)
        # plotX = worldY, plotY = worldX
        self.road_item.setData(road[:, 1], road[:, 0], pen=pg.mkPen((90, 90, 90), width=2))

        for it in self.lanes_items:
            self.plot.removeItem(it)
        self.lanes_items.clear()

        for yc in lane_centers:
            it = pg.PlotDataItem([yc, yc], [x0, x1],
                                 pen=pg.mkPen((60, 60, 60), width=1, style=QtCore.Qt.DashLine))
            self.plot.addItem(it)
            self.lanes_items.append(it)

        # --- vehicles (all participant types) ---
        # Color coding: car=gray, truck=white, twowheeler=yellow, bicycle=green, pedestrian=magenta
        label_colors = {
            "car": (200, 200, 200, 160),
            "truck": (255, 255, 255, 180),
            "twowheeler": (255, 220, 100, 180),
            "bicycle": (100, 255, 100, 180),
            "pedestrian": (255, 100, 255, 180),
        }
        veh_spots = []
        for o in objects:
            if o["label"] not in label_colors:
                continue
            color = label_colors.get(o["label"], (200, 200, 200, 160))
            veh_spots.append({
                "pos": (float(o["pos"][1]), float(o["pos"][0])),  # (Y, X)
                "brush": pg.mkBrush(*color),
                "pen": None
            })
        self.veh_scatter.setData(veh_spots)

        # --- detections ---
        if show_dets and len(dets) > 0:
            vals = np.array([float(d.rcs_dbm) for d in dets], dtype=float)
            vmin, vmax = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
            rgba = rcs_to_rgba(vals, vmin, vmax)

            det_spots = []
            for i, d in enumerate(dets):
                c = rgba[i]
                det_spots.append({
                    "pos": (float(d.y_w), float(d.x_w)),  # (Y, X)
                    "brush": pg.mkBrush(int(c[0]*255), int(c[1]*255), int(c[2]*255), 220),
                    "pen": None
                })
            self.det_scatter.setData(det_spots)
        else:
            self.det_scatter.setData([])

        # --- bboxes ---
        all_labels = ("car", "truck", "twowheeler", "bicycle", "pedestrian")
        existing_pids = {o["participant_id"] for o in objects if o["label"] in all_labels}
        for pid in list(self.bbox_curves.keys()):
            if pid not in existing_pids:
                self.plot.removeItem(self.bbox_curves[pid])
                del self.bbox_curves[pid]
        for pid in list(self.bbox_labels.keys()):
            if pid not in existing_pids:
                self.plot.removeItem(self.bbox_labels[pid])
                del self.bbox_labels[pid]

        # Build lookup for classification confidence by position
        classification_lookup = {}
        if class_results is not None:
            for c in class_results.get('classifications', []):
                pos = c['position']
                classification_lookup[tuple(pos)] = {
                    'class': c['final_class'],
                    'confidence': c['final_confidence']
                }

        # Label colors for text
        label_text_colors = {
            "car": (200, 200, 200),
            "truck": (255, 255, 255),
            "twowheeler": (255, 220, 100),
            "bicycle": (100, 255, 100),
            "pedestrian": (255, 100, 255),
        }

        if show_bbox:
            for o in objects:
                if o["label"] not in all_labels:
                    continue
                pid = o["participant_id"]
                bbox = o["bbox_world"]
                if bbox is None:
                    if pid in self.bbox_curves:
                        self.plot.removeItem(self.bbox_curves[pid])
                        del self.bbox_curves[pid]
                    if pid in self.bbox_labels:
                        self.plot.removeItem(self.bbox_labels[pid])
                        del self.bbox_labels[pid]
                    continue
                poly = bbox_polyline_world(bbox)
                if poly is None:
                    continue
                if pid not in self.bbox_curves:
                    cur = pg.PlotDataItem(pen=pg.mkPen((80, 170, 255), width=2))
                    self.plot.addItem(cur)
                    self.bbox_curves[pid] = cur
                # poly[:,1]=worldY => plotX ; poly[:,0]=worldX => plotY
                self.bbox_curves[pid].setData(poly[:, 1], poly[:, 0])

                # --- Add label text on bounding box (only when show_classification is True) ---
                if show_classification:
                    obj_pos = o["pos"]
                    obj_label = o["label"]
                    
                    # Try to find classification confidence for this object
                    confidence = None
                    classified_type = obj_label  # default to ground truth label
                    for class_pos, class_info in classification_lookup.items():
                        # Match by position proximity
                        if abs(class_pos[0] - obj_pos[0]) < 3.0 and abs(class_pos[1] - obj_pos[1]) < 3.0:
                            classified_type = class_info['class']
                            confidence = class_info['confidence']
                            break
                    
                    # Format label text: "TYPE Conf%"
                    if confidence is not None:
                        label_text = f"{classified_type.upper()} {confidence*100:.0f}%"
                    else:
                        label_text = f"{obj_label.upper()}"
                    
                    # Position at top of bounding box
                    # BEV: plotX = worldY, plotY = worldX
                    label_x = (bbox["ymin"] + bbox["ymax"]) / 2.0  # center in Y (lateral)
                    label_y = bbox["xmax"] + 0.5  # top of bbox in X (forward) + small offset
                    
                    text_color = label_text_colors.get(obj_label, (200, 200, 200))
                    
                    if pid not in self.bbox_labels:
                        txt = pg.TextItem(text=label_text, color=text_color, anchor=(0.5, 1.0))
                        txt.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))
                        self.plot.addItem(txt)
                        self.bbox_labels[pid] = txt
                    else:
                        self.bbox_labels[pid].setText(label_text)
                        self.bbox_labels[pid].setColor(text_color)
                    
                    self.bbox_labels[pid].setPos(label_x, label_y)
                else:
                    # Hide classification labels when show_classification is off
                    if pid in self.bbox_labels:
                        self.plot.removeItem(self.bbox_labels[pid])
                        del self.bbox_labels[pid]
        else:
            # Hide all bbox labels when bbox display is off
            for pid in list(self.bbox_labels.keys()):
                self.plot.removeItem(self.bbox_labels[pid])
            self.bbox_labels.clear()

        # --- HUD metrics ---
        radar_pos = radar["pos"]
        cars = sum(1 for o in objects if o["label"] == "car")
        trucks = sum(1 for o in objects if o["label"] == "truck")
        twowheelers = sum(1 for o in objects if o["label"] == "twowheeler")
        bicycles = sum(1 for o in objects if o["label"] == "bicycle")
        pedestrians = sum(1 for o in objects if o["label"] == "pedestrian")

        lines = []
        lines.append(f"Objects: car={cars}  truck={trucks}  twowheeler={twowheelers}  bicycle={bicycles}  pedestrian={pedestrians}")
        lines.append(f"Radar: x={radar_pos[0]:.1f}  y={radar_pos[1]:.1f}  z={radar_pos[2]:.1f}  yaw={rad2deg(radar['yaw']):.1f}°")

        # per-object speed/range/az (from radar)
        xr, yr = float(radar_pos[0]), float(radar_pos[1])
        for o in objects:
            if o["label"] not in all_labels:
                continue
            px, py = float(o["pos"][0]), float(o["pos"][1])
            vx = float(o["pos"][0]*0)  # placeholder, we compute speed from participant later if needed

            dx = px - xr
            dy = py - yr
            rng = math.hypot(dx, dy)
            az = math.degrees(math.atan2(dy, dx))
            # We do have speedX in engine vel but not in frame object; show speed from bbox/pos isn't reliable.
            # Use pr_eff if present.
            pr = o.get("pr_eff_dbm", None)
            pr_txt = "-" if pr is None else f"{pr:.1f}"
            
            # Detection count for this object
            det_count = o.get("detection_count", 0)

            # If you want exact speed, I can add vel into frame["objects"] in radar_engine.py.
            lines.append(f"PID {o['participant_id']} {o['label']}  R={rng:.1f}m  az={az:.1f}°  dets={det_count}  PrEff={pr_txt} dBm")

        self.hud.setText("\n".join(lines))
        # place HUD at top-left of current view
        vb = self.plot.getViewBox()
        (xmin, xmax), (ymin, ymax) = vb.viewRange()
        self.hud.setPos(xmin + 0.02*(xmax-xmin), ymax - 0.02*(ymax-ymin))

