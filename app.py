# app.py
# Main UI (no nearest-detection plots). Uses ui_components.py for helpers and BEV.

import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from radar_engine import SimEngine, deg2rad, rad2deg
from radar_io import UdpConfig, UdpSender, CsvLogger, ObjectFeatureLogger, detections_to_rows, objects_to_feature_rows
from ui_components import (
    BirdsEyeWindow,
    rcs_to_rgba,
    make_cuboid_mesh,
    make_cuboid_wireframe,
    set_item_pose,
    bbox_polyline_world,
    build_fov_sector_mesh,
    LABEL_COLORS,
    LABEL_EDGE_COLORS,
)
from classification import ClassificationPipeline


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        pg.setConfigOptions(antialias=True)

        self.setWindowTitle("Traffic Radar Simulator (Single Radar) — Refined POC")
        self.resize(1550, 930)

        self.engine = SimEngine(seed=42)

        self.running = False
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(self.engine.dt * 1000))
        self.timer.timeout.connect(self.on_tick)

        self.udp_cfg = UdpConfig(enabled=False, ip="127.0.0.1", port=5005, as_json=True)
        self.udp = UdpSender(self.udp_cfg)
        self.logger = CsvLogger()
        self.object_logger = ObjectFeatureLogger()

        self.bird = BirdsEyeWindow()
        self.bird_visible = False

        # Classification pipeline
        self.classification_pipeline = ClassificationPipeline(
            eps=2.0,
            min_samples=2,
            classifier_type='rule_based'
        )

        # --- layout ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left panel - control tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabPosition(QtWidgets.QTabWidget.West)
        self.tabs.setMinimumWidth(470)
        layout.addWidget(self.tabs, 0)

        # Center - 3D view
        self.view3d = gl.GLViewWidget()
        layout.addWidget(self.view3d, 1)

        # Right panel - Classification
        self.right_panel = self._build_classification_panel()
        layout.addWidget(self.right_panel, 0)

        self._build_toolbar()
        self._build_tabs()

        # visuals
        self.vehicle_mesh = {}      # pid -> mesh item (transparent)
        self.vehicle_edges = {}     # pid -> wireframe line item
        self.bbox_items = {}

        self.road_lines = []
        self.lane_lines = []

        # FOV visuals
        self.fov_mesh = None
        self.fov_outline_outer = None
        self.fov_outline_inner = None
        
        # Create GL items
        self._init_scene()
        
        # setup
        self.apply_world_config()
        self.apply_radar_config()
        self.reset_camera_to_radar(force=True)

        # default objects
        self.engine.add_participant("car", "incoming", 12.0, self.engine.world.lane_centers()[0], loop=True)
        self.refresh_participant_table()

    # ---------------- Toolbar ----------------
    def _build_toolbar(self):
        tb = QtWidgets.QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addAction("Start").triggered.connect(self.start)
        tb.addAction("Pause").triggered.connect(self.pause)
        tb.addAction("Reset").triggered.connect(self.reset_all)
        tb.addSeparator()
        tb.addAction("Reset Camera to Radar").triggered.connect(lambda: self.reset_camera_to_radar(force=True))
        tb.addAction("Bird's Eye Window").triggered.connect(self.toggle_bird)
        tb.addSeparator()

        self.lbl = QtWidgets.QLabel("t=0.0s | dets=0")
        tb.addWidget(self.lbl)

    # ---------------- Tabs ----------------
    def _build_tabs(self):
        self._tab_world()
        self._tab_radar()
        self._tab_traffic()
        self._tab_participant_props()
        self._tab_io()
        self._tab_view()

    def _tab_world(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)

        self.world_xmin = QtWidgets.QDoubleSpinBox(); self.world_xmin.setRange(-1000, 1000); self.world_xmin.setValue(self.engine.world.x_min)
        self.world_xmax = QtWidgets.QDoubleSpinBox(); self.world_xmax.setRange(-1000, 1000); self.world_xmax.setValue(self.engine.world.x_max)
        self.world_lanes = QtWidgets.QComboBox(); self.world_lanes.addItems(["2", "4", "6"]); self.world_lanes.setCurrentText(str(self.engine.world.lanes))
        self.world_lane_w = QtWidgets.QDoubleSpinBox(); self.world_lane_w.setRange(2.5, 5.0); self.world_lane_w.setValue(self.engine.world.lane_width); self.world_lane_w.setDecimals(2)

        f.addRow("X min (m)", self.world_xmin)
        f.addRow("X max (m)", self.world_xmax)
        f.addRow("Lanes", self.world_lanes)
        f.addRow("Lane width (m)", self.world_lane_w)

        btn = QtWidgets.QPushButton("Apply World Config")
        btn.clicked.connect(self.apply_world_config)
        f.addRow(btn)

        self.tabs.addTab(w, "World")

    def _tab_radar(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)

        self.r_x = QtWidgets.QDoubleSpinBox(); self.r_x.setRange(-1000, 1000); self.r_x.setValue(0.0)
        self.r_y = QtWidgets.QDoubleSpinBox(); self.r_y.setRange(-1000, 1000); self.r_y.setValue(0.0)
        self.r_z = QtWidgets.QDoubleSpinBox(); self.r_z.setRange(0, 50); self.r_z.setValue(2.0)
        self.r_yaw = QtWidgets.QDoubleSpinBox(); self.r_yaw.setRange(-180, 180); self.r_yaw.setValue(0.0)
        self.r_fov = QtWidgets.QDoubleSpinBox(); self.r_fov.setRange(1, 180); self.r_fov.setValue(120.0)
        self.r_rmin = QtWidgets.QDoubleSpinBox(); self.r_rmin.setRange(0.5, 50); self.r_rmin.setValue(self.engine.radar.r_min)
        self.r_rmax = QtWidgets.QDoubleSpinBox(); self.r_rmax.setRange(10, 5000); self.r_rmax.setValue(self.engine.radar.r_max)

        # noise
        self.n_r = QtWidgets.QDoubleSpinBox(); self.n_r.setRange(0.0, 10.0); self.n_r.setValue(self.engine.radar.sigma_r); self.n_r.setDecimals(3)
        self.n_az = QtWidgets.QDoubleSpinBox(); self.n_az.setRange(0.0, 10.0); self.n_az.setValue(rad2deg(self.engine.radar.sigma_az)); self.n_az.setDecimals(3)
        self.n_vr = QtWidgets.QDoubleSpinBox(); self.n_vr.setRange(0.0, 10.0); self.n_vr.setValue(self.engine.radar.sigma_vr); self.n_vr.setDecimals(3)

        # equation calibration knob (optional but useful)
        self.k_db = QtWidgets.QDoubleSpinBox(); self.k_db.setRange(0, 200); self.k_db.setValue(self.engine.pr_K_db); self.k_db.setDecimals(2)

        f.addRow("Radar X (m)", self.r_x)
        f.addRow("Radar Y (m)", self.r_y)
        f.addRow("Radar Z (m)", self.r_z)
        f.addRow("Radar Yaw (deg)", self.r_yaw)
        f.addRow("FOV Az (deg)", self.r_fov)
        f.addRow("Range Min (m)", self.r_rmin)
        f.addRow("Range Max (m)", self.r_rmax)
        f.addRow("σ Range (m)", self.n_r)
        f.addRow("σ Azimuth (deg)", self.n_az)
        f.addRow("σ Vr (m/s)", self.n_vr)
        f.addRow("Radar Eqn K (dB)", self.k_db)

        btn = QtWidgets.QPushButton("Apply Radar Config")
        btn.clicked.connect(self.apply_radar_config)
        f.addRow(btn)

        self.tabs.addTab(w, "Radar")

    def _tab_traffic(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        add_box = QtWidgets.QGroupBox("Add Participant")
        f = QtWidgets.QFormLayout(add_box)

        self.add_type = QtWidgets.QComboBox(); self.add_type.addItems(["car", "truck", "twowheeler", "bicycle", "pedestrian"])
        self.add_dir = QtWidgets.QComboBox(); self.add_dir.addItems(["incoming", "outgoing"])
        self.add_speed = QtWidgets.QDoubleSpinBox(); self.add_speed.setRange(0, 60); self.add_speed.setValue(12.0); self.add_speed.setSuffix(" m/s")
        self.add_lane = QtWidgets.QComboBox()
        self._refresh_lane_dropdown()

        self.add_loop = QtWidgets.QCheckBox("Loop (continuous)")
        self.add_loop.setChecked(True)

        f.addRow("Type", self.add_type)
        f.addRow("Direction", self.add_dir)
        f.addRow("Speed", self.add_speed)
        f.addRow("Lane center Y", self.add_lane)
        f.addRow(self.add_loop)

        btn_add = QtWidgets.QPushButton("Add")
        btn_add.clicked.connect(self.add_participant)
        f.addRow(btn_add)

        rand_box = QtWidgets.QGroupBox("Random Traffic Mode")
        fr = QtWidgets.QFormLayout(rand_box)
        self.rand_enable = QtWidgets.QCheckBox("Enable random mode (max 2 in scene)")
        self.rand_enable.stateChanged.connect(self.on_random_toggle)
        fr.addRow(self.rand_enable)

        list_box = QtWidgets.QGroupBox("Active Participants (effective power uses corner combination)")
        lv = QtWidgets.QVBoxLayout(list_box)

        # Updated columns: includes sigma, Pr_eff, and detection count
        self.table = QtWidgets.QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(["PID", "Type", "Dir", "Loop", "LaneY", "SpeedX", "Sigma(m²)", "Dets", "PrEff(dBm)", "Remove"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.itemSelectionChanged.connect(self.on_table_select)
        lv.addWidget(self.table)

        btn_clear = QtWidgets.QPushButton("Clear All")
        btn_clear.clicked.connect(self.clear_participants)
        lv.addWidget(btn_clear)

        v.addWidget(add_box)
        v.addWidget(rand_box)
        v.addWidget(list_box)
        self.tabs.addTab(w, "Traffic")

    def _tab_participant_props(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)

        self.sel_pid = QtWidgets.QLabel("-")
        self.p_len = QtWidgets.QDoubleSpinBox(); self.p_len.setRange(1.0, 30.0); self.p_len.setDecimals(2)
        self.p_wid = QtWidgets.QDoubleSpinBox(); self.p_wid.setRange(1.0, 10.0); self.p_wid.setDecimals(2)
        self.p_hgt = QtWidgets.QDoubleSpinBox(); self.p_hgt.setRange(0.5, 8.0); self.p_hgt.setDecimals(2)
        self.p_sigma = QtWidgets.QDoubleSpinBox(); self.p_sigma.setRange(0.1, 500.0); self.p_sigma.setDecimals(2)

        f.addRow("Selected PID", self.sel_pid)
        f.addRow("Length L (m)", self.p_len)
        f.addRow("Width W (m)", self.p_wid)
        f.addRow("Height H (m)", self.p_hgt)
        f.addRow("Total Sigma (m²)", self.p_sigma)

        btn_apply = QtWidgets.QPushButton("Apply to Selected Participant")
        btn_apply.clicked.connect(self.apply_participant_props)
        f.addRow(btn_apply)

        self.tabs.addTab(w, "Participant Props")

    def _tab_io(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        udp_box = QtWidgets.QGroupBox("UDP Output")
        f = QtWidgets.QFormLayout(udp_box)

        self.udp_enable = QtWidgets.QCheckBox("Enable UDP streaming")
        self.udp_ip = QtWidgets.QLineEdit(self.udp_cfg.ip)
        self.udp_port = QtWidgets.QSpinBox(); self.udp_port.setRange(1, 65535); self.udp_port.setValue(self.udp_cfg.port)
        self.udp_json = QtWidgets.QCheckBox("Send JSON per detection (else CSV line)")
        self.udp_json.setChecked(True)

        f.addRow(self.udp_enable)
        f.addRow("IP", self.udp_ip)
        f.addRow("Port", self.udp_port)
        f.addRow(self.udp_json)

        log_box = QtWidgets.QGroupBox("Detection Logging")
        fl = QtWidgets.QFormLayout(log_box)

        self.log_enable = QtWidgets.QCheckBox("Enable CSV/TXT logging (detections)")
        self.log_path = QtWidgets.QLineEdit("logs/radar_output.csv")
        self.btn_browse = QtWidgets.QPushButton("Browse…")
        self.btn_browse.clicked.connect(self.browse_logfile)

        fl.addRow(self.log_enable)
        fl.addRow("Log file (.csv/.txt)", self.log_path)
        fl.addRow(self.btn_browse)

        obj_log_box = QtWidgets.QGroupBox("Object Feature Logging (per cycle)")
        ofl = QtWidgets.QFormLayout(obj_log_box)

        self.obj_log_enable = QtWidgets.QCheckBox("Enable object feature logging")
        self.obj_log_path = QtWidgets.QLineEdit("logs/object_features.csv")
        self.btn_obj_browse = QtWidgets.QPushButton("Browse…")
        self.btn_obj_browse.clicked.connect(self.browse_obj_logfile)

        ofl.addRow(self.obj_log_enable)
        ofl.addRow("Object log file (.csv)", self.obj_log_path)
        ofl.addRow(self.btn_obj_browse)

        btn_apply = QtWidgets.QPushButton("Apply IO Settings")
        btn_apply.clicked.connect(self.apply_io_config)

        v.addWidget(udp_box)
        v.addWidget(log_box)
        v.addWidget(obj_log_box)
        v.addWidget(btn_apply)
        v.addStretch(1)

        self.tabs.addTab(w, "IO")

    def _tab_view(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)

        self.show_dets = QtWidgets.QCheckBox("Show detections (colored by received dBm)")
        self.show_dets.setChecked(True)

        self.show_bbox = QtWidgets.QCheckBox("Show bbox (light blue)")
        self.show_bbox.setChecked(True)

        self.show_fov = QtWidgets.QCheckBox("Show FOV sector (translucent)")
        self.show_fov.setChecked(True)

        self.lock_camera = QtWidgets.QCheckBox("Lock camera behind radar")
        self.lock_camera.setChecked(True)

        self.show_clutter = QtWidgets.QCheckBox("Show guardrail/clutter detections")
        self.show_clutter.setChecked(True)

        f.addRow(self.show_dets)
        f.addRow(self.show_bbox)
        f.addRow(self.show_fov)
        f.addRow(self.show_clutter)
        f.addRow(self.lock_camera)

        self.tabs.addTab(w, "View")

    # ---------------- Classification Panel (Right Side) ----------------
    def _build_classification_panel(self) -> QtWidgets.QWidget:
        """Build the right-side classification panel."""
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(420)
        panel.setMaximumWidth(480)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Title
        title = QtWidgets.QLabel("Object Classification")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # --- DBSCAN Clustering Controls ---
        dbscan_box = QtWidgets.QGroupBox("1. DBSCAN Clustering")
        dbscan_layout = QtWidgets.QFormLayout(dbscan_box)

        self.dbscan_eps = QtWidgets.QDoubleSpinBox()
        self.dbscan_eps.setRange(0.5, 10.0)
        self.dbscan_eps.setValue(2.0)
        self.dbscan_eps.setSingleStep(0.5)
        self.dbscan_eps.setDecimals(1)
        self.dbscan_eps.setSuffix(" m")
        self.dbscan_eps.valueChanged.connect(self.on_dbscan_params_changed)

        self.dbscan_min_samples = QtWidgets.QSpinBox()
        self.dbscan_min_samples.setRange(1, 10)
        self.dbscan_min_samples.setValue(2)
        self.dbscan_min_samples.valueChanged.connect(self.on_dbscan_params_changed)

        self.show_clusters = QtWidgets.QCheckBox("Show cluster boundaries")
        self.show_clusters.setChecked(True)

        self.show_centroids = QtWidgets.QCheckBox("Show cluster centroids")
        self.show_centroids.setChecked(True)

        dbscan_layout.addRow("Epsilon (ε):", self.dbscan_eps)
        dbscan_layout.addRow("Min Samples:", self.dbscan_min_samples)
        dbscan_layout.addRow(self.show_clusters)
        dbscan_layout.addRow(self.show_centroids)

        # Cluster stats display
        self.cluster_stats_label = QtWidgets.QLabel("Clusters: 0 | Noise: 0")
        dbscan_layout.addRow(self.cluster_stats_label)

        layout.addWidget(dbscan_box)

        # --- Tracking Controls ---
        tracking_box = QtWidgets.QGroupBox("2. Continuous Tracking")
        tracking_layout = QtWidgets.QFormLayout(tracking_box)

        self.tracking_enabled = QtWidgets.QCheckBox("Enable tracking")
        self.tracking_enabled.setChecked(True)

        self.show_track_ids = QtWidgets.QCheckBox("Show track IDs")
        self.show_track_ids.setChecked(True)

        self.show_track_trails = QtWidgets.QCheckBox("Show velocity vectors")
        self.show_track_trails.setChecked(True)

        tracking_layout.addRow(self.tracking_enabled)
        tracking_layout.addRow(self.show_track_ids)
        tracking_layout.addRow(self.show_track_trails)

        # Track stats display
        self.track_stats_label = QtWidgets.QLabel("Active Tracks: 0")
        tracking_layout.addRow(self.track_stats_label)

        layout.addWidget(tracking_box)

        # --- Classification Controls ---
        class_box = QtWidgets.QGroupBox("3. Classification")
        class_layout = QtWidgets.QVBoxLayout(class_box)

        # Classifier selection
        classifier_select_layout = QtWidgets.QHBoxLayout()
        classifier_select_layout.addWidget(QtWidgets.QLabel("Model:"))

        self.classifier_combo = QtWidgets.QComboBox()
        self.classifier_combo.addItems(["Rule-Based", "Naive Bayes"])
        self.classifier_combo.currentIndexChanged.connect(self.on_classifier_changed)
        classifier_select_layout.addWidget(self.classifier_combo)
        classifier_select_layout.addStretch()

        class_layout.addLayout(classifier_select_layout)

        # Show confidence
        self.show_confidence = QtWidgets.QCheckBox("Show confidence scores")
        self.show_confidence.setChecked(True)
        class_layout.addWidget(self.show_confidence)

        # Classification results table
        self.class_table = QtWidgets.QTableWidget(0, 6)
        self.class_table.setHorizontalHeaderLabels([
            "Track", "Class", "Conf%", "Speed", "Points", "Age"
        ])
        self.class_table.horizontalHeader().setStretchLastSection(True)
        self.class_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.class_table.setMaximumHeight(200)
        self.class_table.setAlternatingRowColors(True)
        class_layout.addWidget(self.class_table)

        layout.addWidget(class_box)

        # --- Classification Details ---
        details_box = QtWidgets.QGroupBox("Classification Details")
        details_layout = QtWidgets.QVBoxLayout(details_box)

        self.class_details_text = QtWidgets.QTextEdit()
        self.class_details_text.setReadOnly(True)
        self.class_details_text.setMaximumHeight(180)
        self.class_details_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        details_layout.addWidget(self.class_details_text)

        layout.addWidget(details_box)

        # --- Model Comparison ---
        compare_box = QtWidgets.QGroupBox("Model Comparison")
        compare_layout = QtWidgets.QFormLayout(compare_box)

        self.rule_accuracy_label = QtWidgets.QLabel("Rule-Based: -")
        self.bayes_accuracy_label = QtWidgets.QLabel("Naive Bayes: -")

        compare_layout.addRow(self.rule_accuracy_label)
        compare_layout.addRow(self.bayes_accuracy_label)

        layout.addWidget(compare_box)

        # Reset button
        reset_btn = QtWidgets.QPushButton("Reset Classification Pipeline")
        reset_btn.clicked.connect(self.reset_classification)
        layout.addWidget(reset_btn)

        layout.addStretch(1)
        return panel

    def on_dbscan_params_changed(self):
        """Update DBSCAN parameters."""
        eps = float(self.dbscan_eps.value())
        min_samples = int(self.dbscan_min_samples.value())
        self.classification_pipeline.set_dbscan_params(eps, min_samples)

    def on_classifier_changed(self):
        """Switch classifier type."""
        idx = self.classifier_combo.currentIndex()
        if idx == 0:
            self.classification_pipeline.set_classifier('rule_based')
        else:
            self.classification_pipeline.set_classifier('naive_bayes')

    def reset_classification(self):
        """Reset the classification pipeline."""
        self.classification_pipeline.reset()
        self.class_table.setRowCount(0)
        self.class_details_text.clear()
        self.cluster_stats_label.setText("Clusters: 0 | Noise: 0")
        self.track_stats_label.setText("Active Tracks: 0")

    def update_classification_ui(self, class_results: dict, objects: list):
        """Update the classification panel UI with results."""
        if class_results is None:
            return

        # Update cluster stats
        self.cluster_stats_label.setText(
            f"Clusters: {class_results['num_clusters']} | Noise: {class_results['num_noise']}"
        )

        # Update track stats
        self.track_stats_label.setText(f"Active Tracks: {class_results['num_tracks']}")

        # Update classification table
        classifications = class_results.get('classifications', [])
        self.class_table.setRowCount(len(classifications))

        # Track class colors for visualization
        class_colors = {
            'car': '#B0B0B0',
            'truck': '#FFFFFF',
            'twowheeler': '#FFD700',
            'bicycle': '#00FF00',
            'pedestrian': '#FF00FF',
            'clutter': '#808080',
            'unknown': '#555555'
        }

        for row, c in enumerate(classifications):
            track_id = c['track_id']
            pred_class = c['final_class']
            confidence = c['final_confidence'] * 100
            speed = c['features'].get('speed', 0.0)
            num_points = c['features'].get('num_points', 0)
            age = c['track_age']

            # Track ID
            item = QtWidgets.QTableWidgetItem(f"T{track_id}")
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.class_table.setItem(row, 0, item)

            # Class with color
            item = QtWidgets.QTableWidgetItem(pred_class)
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            color = class_colors.get(pred_class, '#555555')
            item.setForeground(QtGui.QColor(color))
            self.class_table.setItem(row, 1, item)

            # Confidence
            item = QtWidgets.QTableWidgetItem(f"{confidence:.1f}")
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            if confidence >= 70:
                item.setForeground(QtGui.QColor('#00FF00'))
            elif confidence >= 40:
                item.setForeground(QtGui.QColor('#FFFF00'))
            else:
                item.setForeground(QtGui.QColor('#FF6666'))
            self.class_table.setItem(row, 2, item)

            # Speed
            item = QtWidgets.QTableWidgetItem(f"{speed:.1f}")
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.class_table.setItem(row, 3, item)

            # Points
            item = QtWidgets.QTableWidgetItem(f"{int(num_points)}")
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.class_table.setItem(row, 4, item)

            # Age
            item = QtWidgets.QTableWidgetItem(f"{age}")
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.class_table.setItem(row, 5, item)

        self.class_table.resizeColumnsToContents()

        # Update details text
        self._update_classification_details(classifications, objects)

    def _update_classification_details(self, classifications: list, objects: list):
        """Update the detailed classification text view."""
        lines = []

        # Build object lookup by position proximity
        def find_matching_object(pos):
            for o in objects:
                ox, oy = o['pos'][0], o['pos'][1]
                if abs(ox - pos[0]) < 3.0 and abs(oy - pos[1]) < 3.0:
                    return o
            return None

        for c in classifications:
            track_id = c['track_id']
            lines.append(f"═══ Track T{track_id} ═══")

            # Features
            feat = c['features']
            lines.append(f"  Points: {int(feat.get('num_points', 0))}")
            lines.append(f"  Extent: {feat.get('extent_x', 0):.2f}m × {feat.get('extent_y', 0):.2f}m")
            lines.append(f"  RCS: {feat.get('mean_rcs', 0):.1f} ± {feat.get('std_rcs', 0):.1f} dBm")
            lines.append(f"  Speed: {feat.get('speed', 0):.2f} m/s")

            # Rule-based results
            rb = c['rule_based']
            lines.append(f"  Rule-Based: {rb['class']} ({rb['confidence']*100:.1f}%)")

            # Naive Bayes results
            nb = c['naive_bayes']
            lines.append(f"  Naive Bayes: {nb['class']} ({nb['confidence']*100:.1f}%)")

            # Final result
            lines.append(f"  → FINAL: {c['final_class'].upper()} ({c['final_confidence']*100:.1f}%)")

            # Ground truth comparison (if available)
            match_obj = find_matching_object(c['position'])
            if match_obj:
                gt_label = match_obj['label']
                is_correct = (c['final_class'] == gt_label)
                status = "✓" if is_correct else "✗"
                lines.append(f"  Ground Truth: {gt_label} {status}")

            lines.append("")

        # Model comparison summary
        if classifications:
            rule_correct = 0
            bayes_correct = 0
            total = 0

            for c in classifications:
                match_obj = find_matching_object(c['position'])
                if match_obj:
                    total += 1
                    gt = match_obj['label']
                    if c['rule_based']['class'] == gt:
                        rule_correct += 1
                    if c['naive_bayes']['class'] == gt:
                        bayes_correct += 1

            if total > 0:
                rule_acc = (rule_correct / total) * 100
                bayes_acc = (bayes_correct / total) * 100
                self.rule_accuracy_label.setText(f"Rule-Based: {rule_correct}/{total} ({rule_acc:.0f}%)")
                self.bayes_accuracy_label.setText(f"Naive Bayes: {bayes_correct}/{total} ({bayes_acc:.0f}%)")

        self.class_details_text.setText("\n".join(lines))

    # ---------------- Scene init ----------------
    def _init_scene(self):
        self.view3d.setBackgroundColor((0, 0, 0))

        grid = gl.GLGridItem()
        grid.setSize(250, 120)
        grid.setSpacing(5, 5)
        grid.translate(50, 0, 0)
        try:
            grid.setColor((60, 60, 60, 160))
        except Exception:
            pass
        self.view3d.addItem(grid)

        self.radar_line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=3, antialias=True)
        self.view3d.addItem(self.radar_line)

        self.radar_dot = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=10, color=(1, 1, 1, 1))
        self.view3d.addItem(self.radar_dot)

        self.det_scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=5.5)
        self.view3d.addItem(self.det_scatter)

        # Classification visualization items
        self.cluster_boundaries = {}  # cluster_id -> line item
        self.centroid_scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=12, color=(1, 1, 0, 1))
        self.view3d.addItem(self.centroid_scatter)

        self.track_labels = {}  # track_id -> GLTextItem for classification labels
        self.velocity_vectors = {}  # track_id -> line item
        
        # Check if GLTextItem is available (pyqtgraph >= 0.13)
        self._has_gl_text = hasattr(gl, 'GLTextItem')

    def draw_road_and_lanes(self):
        for it in self.road_lines + self.lane_lines:
            try:
                self.view3d.removeItem(it)
            except Exception:
                pass
        self.road_lines.clear()
        self.lane_lines.clear()

        w = self.engine.world
        x0, x1 = w.x_min, w.x_max
        road_w = w.lane_width * w.lanes
        y0, y1 = -road_w / 2.0, road_w / 2.0

        road = np.array([[x0, y0, 0.02], [x1, y0, 0.02], [x1, y1, 0.02], [x0, y1, 0.02], [x0, y0, 0.02]], dtype=float)
        road_item = gl.GLLinePlotItem(pos=road, width=2, antialias=True)
        self.view3d.addItem(road_item)
        self.road_lines.append(road_item)

        for yc in w.lane_centers():
            lane = np.array([[x0, yc, 0.02], [x1, yc, 0.02]], dtype=float)
            li = gl.GLLinePlotItem(pos=lane, width=1, antialias=True)
            self.view3d.addItem(li)
            self.lane_lines.append(li)

    # ---------------- Controls ----------------
    def start(self):
        if not self.running:
            self.running = True
            self.timer.start()
            self.apply_io_config()

    def pause(self):
        self.running = False
        self.timer.stop()

    def reset_all(self):
        self.pause()
        self.engine.reset_time()
        self.det_scatter.setData(pos=np.zeros((0, 3)))
        self.lbl.setText("t=0.0s | dets=0")

    def toggle_bird(self):
        self.bird_visible = not self.bird_visible
        if self.bird_visible:
            self.bird.show()
        else:
            self.bird.hide()

    # ---------------- Apply configs ----------------
    def apply_world_config(self):
        xmin = float(self.world_xmin.value())
        xmax = float(self.world_xmax.value())
        lanes = int(self.world_lanes.currentText())
        lane_w = float(self.world_lane_w.value())
        if xmax <= xmin + 1.0:
            QtWidgets.QMessageBox.warning(self, "Invalid", "X max must be > X min + 1m")
            return

        self.engine.set_world(xmin, xmax, lanes, lane_w)
        self._refresh_lane_dropdown()
        self.draw_road_and_lanes()

    def apply_radar_config(self):
        self.engine.radar.pos = np.array([self.r_x.value(), self.r_y.value(), self.r_z.value()], dtype=float)
        self.engine.radar.yaw = deg2rad(float(self.r_yaw.value()))
        self.engine.radar.fov_az = deg2rad(float(self.r_fov.value()))
        self.engine.radar.r_min = float(self.r_rmin.value())
        self.engine.radar.r_max = float(self.r_rmax.value())

        self.engine.radar.sigma_r = float(self.n_r.value())
        self.engine.radar.sigma_az = deg2rad(float(self.n_az.value()))
        self.engine.radar.sigma_vr = float(self.n_vr.value())

        self.engine.pr_K_db = float(self.k_db.value())

        p = self.engine.radar.pos
        line = np.array([[p[0], p[1], 0.0], [p[0], p[1], p[2]]], dtype=float)
        self.radar_line.setData(pos=line)
        self.radar_dot.setData(pos=np.array([[p[0], p[1], p[2]]], dtype=float))

        self._update_fov_visual()
        self.reset_camera_to_radar(force=True)

    def apply_io_config(self):
        self.udp_cfg.enabled = self.udp_enable.isChecked()
        self.udp_cfg.ip = self.udp_ip.text().strip() or "127.0.0.1"
        self.udp_cfg.port = int(self.udp_port.value())
        self.udp_cfg.as_json = self.udp_json.isChecked()
        self.udp.cfg = self.udp_cfg
        if self.udp_cfg.enabled:
            self.udp.start()
        else:
            self.udp.stop()

        if self.log_enable.isChecked():
            if not self.logger.enabled:
                self.logger.start(self.log_path.text().strip())
        else:
            if self.logger.enabled:
                self.logger.stop()

        # Object feature logging
        if self.obj_log_enable.isChecked():
            if not self.object_logger.enabled:
                self.object_logger.start(self.obj_log_path.text().strip())
        else:
            if self.object_logger.enabled:
                self.object_logger.stop()

    # ---------------- Camera ----------------
    def reset_camera_to_radar(self, force: bool = False):
        if (not force) and (not self.lock_camera.isChecked()):
            return

        p = self.engine.radar.pos
        yaw_deg = rad2deg(self.engine.radar.yaw)

        # Place the "center" ahead of radar so you feel you're looking down-road.
        # This keeps the camera close and stable.
        fwd = np.array([np.cos(self.engine.radar.yaw), np.sin(self.engine.radar.yaw), 0.0], dtype=float)
        center = p + fwd * 25.0
        self.view3d.opts["center"] = QtGui.QVector3D(float(center[0]), float(center[1]), float(p[2]) * 0.3)

        # Behind-radar view
        self.view3d.opts["azimuth"] = 180.0 - yaw_deg
        self.view3d.opts["elevation"] = 12
        self.view3d.opts["distance"] = 45
        self.view3d.update()

    # ---------------- FOV visual ----------------
    def _update_fov_visual(self):
        # remove old
        for it in [self.fov_mesh, self.fov_outline_outer, self.fov_outline_inner]:
            if it is not None:
                try:
                    self.view3d.removeItem(it)
                except Exception:
                    pass
        self.fov_mesh = None
        self.fov_outline_outer = None
        self.fov_outline_inner = None

        verts, faces, (outer, inner) = build_fov_sector_mesh(self.engine.radar, n=50, z=0.025)

        md = gl.MeshData(vertexes=verts, faces=faces)
        mesh = gl.GLMeshItem(meshdata=md, smooth=False, shader="shaded", drawEdges=False, color=(0.2, 0.35, 0.9, 0.12))
        mesh.setGLOptions("translucent")
        self.fov_mesh = mesh
        self.view3d.addItem(mesh)

        out_line = gl.GLLinePlotItem(pos=outer, width=2, antialias=True)
        in_line = gl.GLLinePlotItem(pos=inner, width=2, antialias=True)
        self.fov_outline_outer = out_line
        self.fov_outline_inner = in_line
        self.view3d.addItem(out_line)
        self.view3d.addItem(in_line)

    # ---------------- Traffic ----------------
    def _refresh_lane_dropdown(self):
        self.add_lane.clear()
        for yc in self.engine.world.lane_centers():
            self.add_lane.addItem(f"{yc:.2f}")

    def add_participant(self):
        label = self.add_type.currentText()
        direction = self.add_dir.currentText()
        speed = float(self.add_speed.value())
        lane_y = float(self.add_lane.currentText())
        loop = bool(self.add_loop.isChecked())

        try:
            self.engine.add_participant(label, direction, speed, lane_y, loop=loop)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Cannot spawn", str(e))
            return

        self.refresh_participant_table()


    def clear_participants(self):
        for pid, it in list(self.vehicle_mesh.items()):
            self.view3d.removeItem(it)
        self.vehicle_mesh.clear()

        for pid, it in list(self.vehicle_edges.items()):
            self.view3d.removeItem(it)
        self.vehicle_edges.clear()

        for pid, it in list(self.bbox_items.items()):
            self.view3d.removeItem(it)
        self.bbox_items.clear()

        self.engine.clear_participants()
        self.refresh_participant_table()

    def on_random_toggle(self):
        self.engine.random_mode = self.rand_enable.isChecked()

    # ---------------- Table / participant props ----------------
    def refresh_participant_table(self):
        self.table.setRowCount(0)

        # need latest pr_eff values: render one step without advancing time? We'll show last known.
        # simplest: do nothing; pr_eff updates during sim ticks. But we can show sigma immediately.
        for p in self.engine.participants:
            row = self.table.rowCount()
            self.table.insertRow(row)

            def set_cell(col, text):
                it = QtWidgets.QTableWidgetItem(text)
                it.setFlags(it.flags() ^ QtCore.Qt.ItemIsEditable)
                self.table.setItem(row, col, it)

            set_cell(0, str(p.pid))
            set_cell(1, p.label)
            set_cell(2, p.direction)
            set_cell(3, "Yes" if p.loop else "No")
            set_cell(4, f"{p.lane_y:.2f}")
            set_cell(5, f"{p.vel[0]:.2f}")
            set_cell(6, f"{p.sigma_m2:.2f}")
            set_cell(7, "-")  # detection count - updated during ticks
            set_cell(8, "-")  # pr_eff - updated during ticks

            btn = QtWidgets.QPushButton("Remove")
            btn.clicked.connect(lambda _, pid=p.pid: self.remove_pid(pid))
            self.table.setCellWidget(row, 9, btn)

        self.table.resizeColumnsToContents()

    def update_pr_eff_column(self, objects):
        # called each tick - update both detection count and pr_eff
        pid_to_eff = {o["participant_id"]: o.get("pr_eff_dbm") for o in objects}
        pid_to_dets = {o["participant_id"]: o.get("detection_count", 0) for o in objects}
        for r in range(self.table.rowCount()):
            pid = int(self.table.item(r, 0).text())
            # Detection count (column 7)
            det_count = pid_to_dets.get(pid, 0)
            self.table.item(r, 7).setText(str(det_count))
            # Pr_eff (column 8)
            val = pid_to_eff.get(pid, None)
            txt = "-" if val is None else f"{val:.1f}"
            self.table.item(r, 8).setText(txt)

    def remove_pid(self, pid: int):
        self.engine.remove_participant(pid)
        if pid in self.vehicle_mesh:
            self.view3d.removeItem(self.vehicle_mesh[pid])
            del self.vehicle_mesh[pid]
        if pid in self.vehicle_edges:
            self.view3d.removeItem(self.vehicle_edges[pid])
            del self.vehicle_edges[pid]
        if pid in self.bbox_items:
            self.view3d.removeItem(self.bbox_items[pid])
            del self.bbox_items[pid]
        self.refresh_participant_table()

    def on_table_select(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            self.sel_pid.setText("-")
            return
        row = sel[0].row()
        pid = int(self.table.item(row, 0).text())
        self.load_participant_props(pid)

    def load_participant_props(self, pid: int):
        p = self.engine.get_participant(pid)
        if not p:
            return
        self.sel_pid.setText(str(pid))
        self.p_len.setValue(float(p.L))
        self.p_wid.setValue(float(p.W))
        self.p_hgt.setValue(float(p.H))
        self.p_sigma.setValue(float(p.sigma_m2))

    def apply_participant_props(self):
        pid_txt = self.sel_pid.text().strip()
        if not pid_txt.isdigit():
            return
        pid = int(pid_txt)
        p = self.engine.get_participant(pid)
        if not p:
            return

        p.L = float(self.p_len.value())
        p.W = float(self.p_wid.value())
        p.H = float(self.p_hgt.value())
        p.sigma_m2 = float(self.p_sigma.value())

        if pid in self.vehicle_mesh:
            self.view3d.removeItem(self.vehicle_mesh[pid])
            del self.vehicle_mesh[pid]
        if pid in self.vehicle_edges:
            self.view3d.removeItem(self.vehicle_edges[pid])
            del self.vehicle_edges[pid]

        self.refresh_participant_table()

    # ---------------- IO ----------------
    def browse_logfile(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select log file", self.log_path.text(), "CSV (*.csv);;TXT (*.txt)")
        if path:
            self.log_path.setText(path)

    def browse_obj_logfile(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select object log file", self.obj_log_path.text(), "CSV (*.csv)")
        if path:
            self.obj_log_path.setText(path)

    # ---------------- Tick ----------------
    def on_tick(self):
        # clutter toggle (engine-side)
        self.engine.clutter_enabled = self.show_clutter.isChecked()

        frame = self.engine.step()
        dets = frame["detections"]

        self.lbl.setText(f"t={frame['t_s']:.1f}s | dets={len(dets)}")
        self.reset_camera_to_radar(force=False)

        rows = detections_to_rows(frame)
        self.udp.send_rows(rows)
        self.logger.write_rows(rows)

        # Log object features each cycle
        obj_rows = objects_to_feature_rows(frame)
        self.object_logger.write_rows(obj_rows)

        # update table effective power display
        self.update_pr_eff_column(frame["objects"])

        # Run classification pipeline
        if self.tracking_enabled.isChecked():
            class_results = self.classification_pipeline.process_frame(dets, frame["t_s"])
            self.update_classification_ui(class_results, frame["objects"])
        else:
            class_results = None

        self.update_3d(frame, class_results)

        if self.bird_visible:
            self.bird.update_view(frame, show_bbox=self.show_bbox.isChecked(), show_dets=self.show_dets.isChecked(), class_results=class_results)

    def update_3d(self, frame: dict, class_results: dict = None):
        # FOV visibility toggle
        if self.fov_mesh is not None:
            self.fov_mesh.setVisible(self.show_fov.isChecked())
            self.fov_outline_outer.setVisible(self.show_fov.isChecked())
            self.fov_outline_inner.setVisible(self.show_fov.isChecked())

        # vehicles - transparent with wireframe to highlight detections
        for o in frame["objects"]:
            pid = o["participant_id"]
            pos = o["pos"]
            yaw = o["yaw"]
            L, W, H = o["dims"]
            label = o["label"]

            # Get colors for this object type
            mesh_color = LABEL_COLORS.get(label, (0.7, 0.7, 0.7, 0.15))
            edge_color = LABEL_EDGE_COLORS.get(label, (0.8, 0.8, 0.8, 0.8))

            if pid not in self.vehicle_mesh:
                # Create transparent mesh
                md = make_cuboid_mesh(L, W, H)
                mesh = gl.GLMeshItem(
                    meshdata=md, 
                    smooth=False, 
                    computeNormals=True, 
                    shader="shaded", 
                    drawEdges=False,
                    color=mesh_color
                )
                mesh.setGLOptions("translucent")
                self.vehicle_mesh[pid] = mesh
                self.view3d.addItem(mesh)
                
                # Create wireframe edges for visibility
                edge_verts = make_cuboid_wireframe(L, W, H)
                edge_item = gl.GLLinePlotItem(
                    pos=edge_verts,
                    color=edge_color,
                    width=2.0,
                    antialias=True,
                    mode='lines'
                )
                self.vehicle_edges[pid] = edge_item
                self.view3d.addItem(edge_item)

            set_item_pose(self.vehicle_mesh[pid], pos, yaw)
            set_item_pose(self.vehicle_edges[pid], pos, yaw)

        current = {o["participant_id"] for o in frame["objects"]}
        for pid in list(self.vehicle_mesh.keys()):
            if pid not in current:
                self.view3d.removeItem(self.vehicle_mesh[pid])
                del self.vehicle_mesh[pid]
                if pid in self.vehicle_edges:
                    self.view3d.removeItem(self.vehicle_edges[pid])
                    del self.vehicle_edges[pid]

        # detections: color by received rcs_dbm - prominent display
        if self.show_dets.isChecked() and frame["detections"]:
            pts = []
            vals = []
            for d in frame["detections"]:
                # Elevate detections slightly so they're visible through transparent objects
                pts.append([d.x_w, d.y_w, 0.5])
                vals.append(float(d.rcs_dbm))
            pts = np.array(pts, dtype=float)
            vals = np.array(vals, dtype=float)

            vmin = float(np.percentile(vals, 5))
            vmax = float(np.percentile(vals, 95))
            colors = rcs_to_rgba(vals, vmin, vmax)

            # Larger detection points to be clearly visible through transparent objects
            self.det_scatter.setData(pos=pts, size=8.0, color=colors)
        else:
            self.det_scatter.setData(pos=np.zeros((0, 3)))

        # bboxes
        for o in frame["objects"]:
            pid = o["participant_id"]
            bbox = o["bbox_world"]
            if (not self.show_bbox.isChecked()) or bbox is None:
                if pid in self.bbox_items:
                    self.view3d.removeItem(self.bbox_items[pid])
                    del self.bbox_items[pid]
                continue

            poly = bbox_polyline_world(bbox)
            if poly is None:
                continue

            if pid not in self.bbox_items:
                it = gl.GLLinePlotItem(pos=poly, width=2, antialias=True)
                self.bbox_items[pid] = it
                self.view3d.addItem(it)
            else:
                self.bbox_items[pid].setData(pos=poly)

        # Classification visualization
        self._update_classification_visuals(class_results)

    def _update_classification_visuals(self, class_results: dict):
        """Update 3D visualization of clusters, tracks, and classification labels."""
        if class_results is None:
            # Hide all classification visuals
            self.centroid_scatter.setData(pos=np.zeros((0, 3)))
            for cid in list(self.cluster_boundaries.keys()):
                self.view3d.removeItem(self.cluster_boundaries[cid])
            self.cluster_boundaries.clear()
            for tid in list(self.velocity_vectors.keys()):
                self.view3d.removeItem(self.velocity_vectors[tid])
            self.velocity_vectors.clear()
            for tid in list(self.track_labels.keys()):
                self.view3d.removeItem(self.track_labels[tid])
            self.track_labels.clear()
            return

        # Class colors for visualization
        class_colors = {
            'car': (0.7, 0.7, 0.7, 1.0),
            'truck': (1.0, 1.0, 1.0, 1.0),
            'twowheeler': (1.0, 0.85, 0.0, 1.0),
            'bicycle': (0.0, 1.0, 0.0, 1.0),
            'pedestrian': (1.0, 0.0, 1.0, 1.0),
            'clutter': (0.5, 0.5, 0.5, 1.0),
            'unknown': (0.3, 0.3, 0.3, 1.0)
        }

        tracks = class_results.get('tracks', [])
        clusters = class_results.get('clusters', [])
        classifications = class_results.get('classifications', [])

        # Build lookup for classification by track_id
        class_by_track = {}
        for c in classifications:
            class_by_track[c['track_id']] = c

        # Update cluster boundaries
        current_cluster_ids = set()
        if self.show_clusters.isChecked():
            for cluster in clusters:
                cid = cluster.cluster_id
                current_cluster_ids.add(cid)

                # Create boundary polygon from cluster detections
                if cluster.detections:
                    xs = [d.x_w for d in cluster.detections]
                    ys = [d.y_w for d in cluster.detections]

                    # Create convex hull-like boundary (simple approach: rectangle)
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                    z = 0.6

                    # Add small padding
                    pad = 0.3
                    pts = np.array([
                        [xmin - pad, ymin - pad, z],
                        [xmax + pad, ymin - pad, z],
                        [xmax + pad, ymax + pad, z],
                        [xmin - pad, ymax + pad, z],
                        [xmin - pad, ymin - pad, z],
                    ], dtype=float)

                    if cid not in self.cluster_boundaries:
                        line = gl.GLLinePlotItem(pos=pts, width=2, antialias=True, color=(0.2, 0.8, 1.0, 0.7))
                        self.cluster_boundaries[cid] = line
                        self.view3d.addItem(line)
                    else:
                        self.cluster_boundaries[cid].setData(pos=pts)

        # Remove old cluster boundaries
        for cid in list(self.cluster_boundaries.keys()):
            if cid not in current_cluster_ids:
                self.view3d.removeItem(self.cluster_boundaries[cid])
                del self.cluster_boundaries[cid]

        # Update centroids
        if self.show_centroids.isChecked() and tracks:
            centroid_pts = []
            centroid_colors = []
            for track in tracks:
                pos = track.pos
                centroid_pts.append([pos[0], pos[1], 0.8])
                color = class_colors.get(track.predicted_class, (0.5, 0.5, 0.5, 1.0))
                centroid_colors.append(color)

            centroid_pts = np.array(centroid_pts, dtype=float)
            centroid_colors = np.array(centroid_colors, dtype=float)
            self.centroid_scatter.setData(pos=centroid_pts, size=14, color=centroid_colors)
        else:
            self.centroid_scatter.setData(pos=np.zeros((0, 3)))

        # Update velocity vectors and track labels
        current_track_ids = set()
        for track in tracks:
            tid = track.track_id
            current_track_ids.add(tid)

            pos = track.pos
            vel = track.vel
            speed = float(np.linalg.norm(vel))
            color = class_colors.get(track.predicted_class, (0.5, 0.5, 0.5, 1.0))

            # Velocity vectors
            if self.show_track_trails.isChecked():
                if speed > 0.5:  # Only show if moving
                    # Scale velocity for visualization
                    scale = 0.5
                    vel_pts = np.array([
                        [pos[0], pos[1], 0.9],
                        [pos[0] + vel[0] * scale, pos[1] + vel[1] * scale, 0.9]
                    ], dtype=float)

                    if tid not in self.velocity_vectors:
                        line = gl.GLLinePlotItem(pos=vel_pts, width=3, antialias=True, color=color)
                        self.velocity_vectors[tid] = line
                        self.view3d.addItem(line)
                    else:
                        self.velocity_vectors[tid].setData(pos=vel_pts, color=color)
                else:
                    if tid in self.velocity_vectors:
                        self.view3d.removeItem(self.velocity_vectors[tid])
                        del self.velocity_vectors[tid]

            # Classification labels in 3D view
            if self.show_track_ids.isChecked():
                # Get classification info for this track
                class_info = class_by_track.get(tid)
                if class_info:
                    pred_class = class_info['final_class']
                    confidence = class_info['final_confidence'] * 100
                    label_text = f"{pred_class.upper()} {confidence:.0f}%"
                else:
                    pred_class = track.predicted_class
                    label_text = f"T{tid}"

                # Position label above the object
                label_pos = (float(pos[0]), float(pos[1]), 2.5)
                
                # Convert color to QColor format for GLTextItem
                qcolor = QtGui.QColor(
                    int(color[0] * 255),
                    int(color[1] * 255),
                    int(color[2] * 255)
                )

                if self._has_gl_text:
                    # Use GLTextItem if available (pyqtgraph >= 0.13)
                    if tid not in self.track_labels:
                        text_item = gl.GLTextItem(
                            pos=label_pos,
                            text=label_text,
                            color=qcolor,
                            font=QtGui.QFont('Arial', 10, QtGui.QFont.Bold)
                        )
                        self.track_labels[tid] = text_item
                        self.view3d.addItem(text_item)
                    else:
                        self.track_labels[tid].setData(pos=label_pos, text=label_text, color=qcolor)
                else:
                    # Fallback: Use vertical line markers with colored tips for older pyqtgraph
                    # This creates a "flag" marker pointing up from the object
                    marker_pts = np.array([
                        [pos[0], pos[1], 1.8],
                        [pos[0], pos[1], 3.0],
                    ], dtype=float)
                    
                    if tid not in self.track_labels:
                        line = gl.GLLinePlotItem(pos=marker_pts, width=4, antialias=True, color=color)
                        self.track_labels[tid] = line
                        self.view3d.addItem(line)
                    else:
                        self.track_labels[tid].setData(pos=marker_pts, color=color)

        # Remove old velocity vectors
        for tid in list(self.velocity_vectors.keys()):
            if tid not in current_track_ids:
                self.view3d.removeItem(self.velocity_vectors[tid])
                del self.velocity_vectors[tid]

        # Remove old track labels
        for tid in list(self.track_labels.keys()):
            if tid not in current_track_ids or not self.show_track_ids.isChecked():
                self.view3d.removeItem(self.track_labels[tid])
                del self.track_labels[tid]

    def closeEvent(self, event):
        try:
            self.udp.stop()
        except Exception:
            pass
        try:
            self.logger.stop()
        except Exception:
            pass
        try:
            self.object_logger.stop()
        except Exception:
            pass
        try:
            self.bird.close()
        except Exception:
            pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
