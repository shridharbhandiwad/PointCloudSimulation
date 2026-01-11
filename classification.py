# classification.py
# DBSCAN clustering, continuous tracking, and classification modules
# Implements Rule-based and Naive Bayes classifiers for radar detections

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import math
import numpy as np
from collections import defaultdict


# =============================================================================
# DBSCAN Clustering
# =============================================================================

@dataclass
class Cluster:
    """Represents a cluster of radar detections."""
    cluster_id: int
    detections: List[Any]  # List of Detection objects
    centroid: np.ndarray   # (x, y) world coordinates
    
    # Cluster features
    num_points: int = 0
    extent_x: float = 0.0  # Width in X direction
    extent_y: float = 0.0  # Width in Y direction
    mean_rcs: float = 0.0  # Mean RCS in dBm
    std_rcs: float = 0.0   # Std of RCS
    mean_vr: float = 0.0   # Mean radial velocity
    std_vr: float = 0.0    # Std of radial velocity
    
    def compute_features(self):
        """Compute cluster features from detections."""
        if not self.detections:
            return
            
        self.num_points = len(self.detections)
        
        xs = [d.x_w for d in self.detections]
        ys = [d.y_w for d in self.detections]
        rcs = [d.rcs_dbm for d in self.detections]
        vrs = [d.vr for d in self.detections]
        
        self.centroid = np.array([np.mean(xs), np.mean(ys)], dtype=float)
        self.extent_x = max(xs) - min(xs) if len(xs) > 1 else 0.1
        self.extent_y = max(ys) - min(ys) if len(ys) > 1 else 0.1
        self.mean_rcs = float(np.mean(rcs))
        self.std_rcs = float(np.std(rcs)) if len(rcs) > 1 else 0.0
        self.mean_vr = float(np.mean(vrs))
        self.std_vr = float(np.std(vrs)) if len(vrs) > 1 else 0.0


class DBSCANClusterer:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    for grouping radar detections into objects.
    """
    
    def __init__(self, eps: float = 2.0, min_samples: int = 2):
        """
        Args:
            eps: Maximum distance between two samples to be considered in the same neighborhood
            min_samples: Minimum number of samples in a neighborhood to form a core point
        """
        self.eps = eps
        self.min_samples = min_samples
        self.next_cluster_id = 1
        
    def _euclidean_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute Euclidean distance between two points."""
        return float(np.linalg.norm(p1 - p2))
    
    def _get_neighbors(self, points: np.ndarray, point_idx: int) -> List[int]:
        """Get indices of all points within eps distance of point_idx."""
        neighbors = []
        for i in range(len(points)):
            if self._euclidean_distance(points[point_idx], points[i]) <= self.eps:
                neighbors.append(i)
        return neighbors
    
    def cluster(self, detections: List[Any]) -> Tuple[List[Cluster], List[Any]]:
        """
        Perform DBSCAN clustering on detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            (clusters, noise_detections): List of Cluster objects and noise detections
        """
        if not detections:
            return [], []
        
        # Extract (x, y) coordinates
        points = np.array([[d.x_w, d.y_w] for d in detections], dtype=float)
        n = len(points)
        
        # Labels: -1 = unvisited, 0 = noise, >0 = cluster ID
        labels = np.full(n, -1, dtype=int)
        cluster_id = 0
        
        for i in range(n):
            if labels[i] != -1:  # Already processed
                continue
                
            neighbors = self._get_neighbors(points, i)
            
            if len(neighbors) < self.min_samples:
                labels[i] = 0  # Mark as noise
            else:
                cluster_id += 1
                self._expand_cluster(points, labels, i, neighbors, cluster_id)
        
        # Group detections by cluster
        clusters = []
        noise = []
        
        cluster_detections = defaultdict(list)
        for i, label in enumerate(labels):
            if label == 0:
                noise.append(detections[i])
            else:
                cluster_detections[label].append(detections[i])
        
        for cid, dets in cluster_detections.items():
            cluster = Cluster(
                cluster_id=self.next_cluster_id,
                detections=dets,
                centroid=np.zeros(2)
            )
            cluster.compute_features()
            clusters.append(cluster)
            self.next_cluster_id += 1
        
        return clusters, noise
    
    def _expand_cluster(self, points: np.ndarray, labels: np.ndarray, 
                        point_idx: int, neighbors: List[int], cluster_id: int):
        """Expand cluster from a core point."""
        labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if labels[neighbor_idx] == 0:  # Was noise, now border point
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == -1:  # Unvisited
                labels[neighbor_idx] = cluster_id
                neighbor_neighbors = self._get_neighbors(points, neighbor_idx)
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors.extend(neighbor_neighbors)
            
            i += 1


# =============================================================================
# Continuous Object Tracking
# =============================================================================

@dataclass
class TrackedObject:
    """Represents a continuously tracked object."""
    track_id: int
    cluster: Optional[Cluster] = None
    
    # State estimation (position, velocity)
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    # Track history
    age: int = 0                    # Number of frames this track has existed
    hits: int = 0                   # Number of successful associations
    misses: int = 0                 # Consecutive frames without association
    last_update_time: float = 0.0
    
    # Classification
    predicted_class: str = "unknown"
    class_confidence: float = 0.0
    class_history: List[Tuple[str, float]] = field(default_factory=list)
    
    # Features history for classification
    feature_history: List[Dict[str, float]] = field(default_factory=list)
    
    def update_from_cluster(self, cluster: Cluster, dt: float):
        """Update track state from associated cluster."""
        new_pos = cluster.centroid.copy()
        
        if self.hits > 0 and dt > 0:
            # Estimate velocity
            self.vel = (new_pos - self.pos) / dt
        
        self.pos = new_pos
        self.cluster = cluster
        self.hits += 1
        self.misses = 0
        self.age += 1
        
        # Store features for classification
        features = {
            'num_points': cluster.num_points,
            'extent_x': cluster.extent_x,
            'extent_y': cluster.extent_y,
            'mean_rcs': cluster.mean_rcs,
            'std_rcs': cluster.std_rcs,
            'mean_vr': cluster.mean_vr,
            'std_vr': cluster.std_vr,
            'speed': float(np.linalg.norm(self.vel))
        }
        self.feature_history.append(features)
        
        # Keep only last N features
        if len(self.feature_history) > 20:
            self.feature_history = self.feature_history[-20:]
    
    def predict(self, dt: float):
        """Predict next state (simple linear motion model)."""
        self.pos = self.pos + self.vel * dt
        self.misses += 1
        self.age += 1
    
    def get_average_features(self) -> Dict[str, float]:
        """Get time-averaged features for more stable classification."""
        if not self.feature_history:
            return {}
        
        avg = {}
        for key in self.feature_history[0].keys():
            vals = [f[key] for f in self.feature_history]
            avg[key] = float(np.mean(vals))
        return avg


class ObjectTracker:
    """
    Multi-object tracker using nearest-neighbor association.
    Maintains persistent object IDs across frames.
    """
    
    def __init__(self, max_misses: int = 5, min_hits: int = 2, 
                 association_threshold: float = 5.0):
        """
        Args:
            max_misses: Maximum consecutive misses before track deletion
            min_hits: Minimum hits before track is considered confirmed
            association_threshold: Maximum distance for cluster-track association
        """
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.association_threshold = association_threshold
        
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 1
        self.last_time = 0.0
    
    def update(self, clusters: List[Cluster], timestamp: float) -> List[TrackedObject]:
        """
        Update tracker with new clusters.
        
        Args:
            clusters: List of clusters from DBSCAN
            timestamp: Current timestamp in seconds
            
        Returns:
            List of active tracked objects
        """
        dt = timestamp - self.last_time if self.last_time > 0 else 0.1
        self.last_time = timestamp
        
        # Predict all existing tracks
        for track in self.tracks.values():
            track.predict(dt)
        
        # Associate clusters to tracks using Hungarian algorithm approximation
        unassigned_clusters = list(range(len(clusters)))
        unassigned_tracks = list(self.tracks.keys())
        
        # Compute cost matrix
        if clusters and self.tracks:
            assignments = self._associate(clusters, unassigned_tracks)
            
            for cluster_idx, track_id in assignments:
                if cluster_idx in unassigned_clusters:
                    unassigned_clusters.remove(cluster_idx)
                if track_id in unassigned_tracks:
                    unassigned_tracks.remove(track_id)
                
                # Update track with associated cluster
                self.tracks[track_id].update_from_cluster(clusters[cluster_idx], dt)
                self.tracks[track_id].last_update_time = timestamp
        
        # Create new tracks for unassigned clusters
        for cluster_idx in unassigned_clusters:
            new_track = TrackedObject(
                track_id=self.next_track_id,
                cluster=clusters[cluster_idx],
                pos=clusters[cluster_idx].centroid.copy(),
                last_update_time=timestamp
            )
            new_track.hits = 1
            new_track.age = 1
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1
        
        # Remove dead tracks
        to_remove = []
        for track_id, track in self.tracks.items():
            if track.misses > self.max_misses:
                to_remove.append(track_id)
        for track_id in to_remove:
            del self.tracks[track_id]
        
        # Return confirmed tracks
        return [t for t in self.tracks.values() if t.hits >= self.min_hits]
    
    def _associate(self, clusters: List[Cluster], track_ids: List[int]) -> List[Tuple[int, int]]:
        """
        Associate clusters to tracks using greedy nearest neighbor.
        Returns list of (cluster_idx, track_id) pairs.
        """
        assignments = []
        
        # Compute distance matrix
        distances = {}
        for ci, cluster in enumerate(clusters):
            for tid in track_ids:
                track = self.tracks[tid]
                dist = float(np.linalg.norm(cluster.centroid - track.pos))
                distances[(ci, tid)] = dist
        
        # Greedy assignment (smallest distance first)
        used_clusters = set()
        used_tracks = set()
        
        sorted_pairs = sorted(distances.items(), key=lambda x: x[1])
        
        for (ci, tid), dist in sorted_pairs:
            if ci in used_clusters or tid in used_tracks:
                continue
            if dist > self.association_threshold:
                continue
            
            assignments.append((ci, tid))
            used_clusters.add(ci)
            used_tracks.add(tid)
        
        return assignments
    
    def get_all_tracks(self) -> List[TrackedObject]:
        """Get all tracks (including tentative ones)."""
        return list(self.tracks.values())
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_track_id = 1
        self.last_time = 0.0


# =============================================================================
# Classification Models
# =============================================================================

class RuleBasedClassifier:
    """
    Rule-based classifier using handcrafted thresholds on radar features.
    Fast and interpretable, works well when feature distributions are known.
    """
    
    CLASSES = ['car', 'truck', 'twowheeler', 'bicycle', 'pedestrian', 'clutter']
    
    def __init__(self):
        # Define rules based on typical radar characteristics
        # Rules are (min, max) ranges for each feature
        self.rules = {
            'truck': {
                'num_points': (25, 200),
                'extent_x': (3.0, 15.0),
                'extent_y': (1.5, 5.0),
                'mean_rcs': (20, 50),
                'speed': (0.5, 30.0)
            },
            'car': {
                'num_points': (10, 50),
                'extent_x': (1.5, 6.0),
                'extent_y': (1.0, 3.0),
                'mean_rcs': (10, 35),
                'speed': (0.5, 40.0)
            },
            'twowheeler': {
                'num_points': (3, 15),
                'extent_x': (0.8, 3.0),
                'extent_y': (0.3, 1.5),
                'mean_rcs': (5, 25),
                'speed': (0.5, 35.0)
            },
            'bicycle': {
                'num_points': (1, 8),
                'extent_x': (0.5, 2.5),
                'extent_y': (0.2, 1.2),
                'mean_rcs': (-5, 20),
                'speed': (0.3, 15.0)
            },
            'pedestrian': {
                'num_points': (1, 6),
                'extent_x': (0.2, 1.5),
                'extent_y': (0.2, 1.0),
                'mean_rcs': (-10, 15),
                'speed': (0.1, 3.0)
            },
            'clutter': {
                'num_points': (1, 5),
                'extent_x': (0.0, 1.0),
                'extent_y': (0.0, 1.0),
                'mean_rcs': (-20, 20),
                'speed': (0.0, 0.5)
            }
        }
    
    def _compute_rule_score(self, features: Dict[str, float], 
                           class_rules: Dict[str, Tuple[float, float]]) -> float:
        """
        Compute how well features match a class's rules.
        Returns score between 0 and 1.
        """
        scores = []
        
        for feature_name, (min_val, max_val) in class_rules.items():
            if feature_name not in features:
                continue
            
            value = features[feature_name]
            
            # Score based on how well value fits in range
            if min_val <= value <= max_val:
                # Perfect fit - compute normalized position
                range_size = max_val - min_val
                if range_size > 0:
                    center = (min_val + max_val) / 2
                    # Higher score for values closer to center
                    dist_from_center = abs(value - center)
                    score = 1.0 - (dist_from_center / (range_size / 2)) * 0.3
                else:
                    score = 1.0
            else:
                # Outside range - penalize based on distance
                if value < min_val:
                    dist = min_val - value
                    range_size = max(1.0, max_val - min_val)
                else:
                    dist = value - max_val
                    range_size = max(1.0, max_val - min_val)
                
                score = max(0.0, 1.0 - dist / range_size)
            
            scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def classify(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify based on rule matching.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            (predicted_class, confidence, all_class_scores)
        """
        if not features:
            return 'unknown', 0.0, {}
        
        class_scores = {}
        for class_name, rules in self.rules.items():
            class_scores[class_name] = self._compute_rule_score(features, rules)
        
        # Find best class
        best_class = max(class_scores, key=class_scores.get)
        best_score = class_scores[best_class]
        
        # Compute confidence (relative to other classes)
        total_score = sum(class_scores.values())
        if total_score > 0:
            confidence = best_score / total_score
        else:
            confidence = 0.0
        
        return best_class, confidence, class_scores


class NaiveBayesClassifier:
    """
    Naive Bayes classifier for radar object classification.
    Assumes features are conditionally independent given the class.
    Uses Gaussian distributions for continuous features.
    """
    
    CLASSES = ['car', 'truck', 'twowheeler', 'bicycle', 'pedestrian', 'clutter']
    
    def __init__(self):
        # Pre-trained parameters (mean, std) for each feature per class
        # Based on typical radar characteristics
        self.params = {
            'car': {
                'num_points': (25.0, 10.0),
                'extent_x': (3.5, 1.0),
                'extent_y': (1.8, 0.5),
                'mean_rcs': (22.0, 8.0),
                'speed': (12.0, 6.0)
            },
            'truck': {
                'num_points': (50.0, 20.0),
                'extent_x': (8.0, 2.5),
                'extent_y': (2.5, 0.6),
                'mean_rcs': (30.0, 8.0),
                'speed': (10.0, 5.0)
            },
            'twowheeler': {
                'num_points': (8.0, 4.0),
                'extent_x': (2.0, 0.6),
                'extent_y': (0.8, 0.3),
                'mean_rcs': (15.0, 7.0),
                'speed': (10.0, 5.0)
            },
            'bicycle': {
                'num_points': (4.0, 2.0),
                'extent_x': (1.5, 0.5),
                'extent_y': (0.6, 0.3),
                'mean_rcs': (8.0, 6.0),
                'speed': (5.0, 3.0)
            },
            'pedestrian': {
                'num_points': (2.5, 1.5),
                'extent_x': (0.6, 0.3),
                'extent_y': (0.5, 0.2),
                'mean_rcs': (5.0, 5.0),
                'speed': (1.5, 0.8)
            },
            'clutter': {
                'num_points': (2.0, 1.0),
                'extent_x': (0.4, 0.3),
                'extent_y': (0.3, 0.2),
                'mean_rcs': (10.0, 8.0),
                'speed': (0.1, 0.1)
            }
        }
        
        # Prior probabilities (can be adjusted based on expected distribution)
        self.priors = {
            'car': 0.35,
            'truck': 0.15,
            'twowheeler': 0.15,
            'bicycle': 0.10,
            'pedestrian': 0.10,
            'clutter': 0.15
        }
    
    def _gaussian_pdf(self, x: float, mean: float, std: float) -> float:
        """Compute Gaussian probability density."""
        std = max(std, 1e-6)  # Avoid division by zero
        exponent = -0.5 * ((x - mean) / std) ** 2
        return (1.0 / (std * math.sqrt(2 * math.pi))) * math.exp(exponent)
    
    def _log_likelihood(self, features: Dict[str, float], class_name: str) -> float:
        """Compute log-likelihood of features given class."""
        if class_name not in self.params:
            return float('-inf')
        
        class_params = self.params[class_name]
        log_prob = math.log(self.priors.get(class_name, 0.1))
        
        for feature_name, value in features.items():
            if feature_name in class_params:
                mean, std = class_params[feature_name]
                prob = self._gaussian_pdf(value, mean, std)
                # Add small epsilon to avoid log(0)
                log_prob += math.log(max(prob, 1e-300))
        
        return log_prob
    
    def classify(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify using Naive Bayes.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            (predicted_class, confidence, all_class_probabilities)
        """
        if not features:
            return 'unknown', 0.0, {}
        
        # Compute log-likelihoods for each class
        log_likelihoods = {}
        for class_name in self.CLASSES:
            log_likelihoods[class_name] = self._log_likelihood(features, class_name)
        
        # Convert to probabilities using log-sum-exp trick for numerical stability
        max_ll = max(log_likelihoods.values())
        
        # Compute unnormalized probabilities
        probs = {}
        for class_name, ll in log_likelihoods.items():
            probs[class_name] = math.exp(ll - max_ll)
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            for class_name in probs:
                probs[class_name] /= total
        
        # Find best class
        best_class = max(probs, key=probs.get)
        confidence = probs[best_class]
        
        return best_class, confidence, probs
    
    def update_params(self, class_name: str, feature_name: str, mean: float, std: float):
        """Update parameters for online learning."""
        if class_name not in self.params:
            self.params[class_name] = {}
        self.params[class_name][feature_name] = (mean, std)


# =============================================================================
# Combined Classification Pipeline
# =============================================================================

class ClassificationPipeline:
    """
    Complete classification pipeline combining clustering, tracking, and classification.
    """
    
    def __init__(self, 
                 eps: float = 2.0,
                 min_samples: int = 2,
                 classifier_type: str = 'rule_based'):
        """
        Args:
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples
            classifier_type: 'rule_based' or 'naive_bayes'
        """
        self.clusterer = DBSCANClusterer(eps=eps, min_samples=min_samples)
        self.tracker = ObjectTracker()
        
        self.rule_classifier = RuleBasedClassifier()
        self.bayes_classifier = NaiveBayesClassifier()
        self.classifier_type = classifier_type
        
        # Results storage
        self.last_clusters: List[Cluster] = []
        self.last_noise: List[Any] = []
        self.last_tracks: List[TrackedObject] = []
        
    def set_dbscan_params(self, eps: float, min_samples: int):
        """Update DBSCAN parameters."""
        self.clusterer.eps = eps
        self.clusterer.min_samples = min_samples
    
    def set_classifier(self, classifier_type: str):
        """Set which classifier to use."""
        self.classifier_type = classifier_type
    
    def process_frame(self, detections: List[Any], timestamp: float) -> Dict[str, Any]:
        """
        Process a frame of detections through the full pipeline.
        
        Args:
            detections: List of Detection objects
            timestamp: Current timestamp in seconds
            
        Returns:
            Dictionary with clustering, tracking, and classification results
        """
        # Filter out clutter/noise detections if they have participant_id == 0
        object_detections = [d for d in detections if d.participant_id != 0]
        clutter_detections = [d for d in detections if d.participant_id == 0]
        
        # Step 1: DBSCAN Clustering
        clusters, noise = self.clusterer.cluster(object_detections)
        self.last_clusters = clusters
        self.last_noise = noise
        
        # Step 2: Object Tracking
        tracks = self.tracker.update(clusters, timestamp)
        self.last_tracks = tracks
        
        # Step 3: Classification for each track
        classification_results = []
        
        for track in tracks:
            features = track.get_average_features()
            
            # Get classification from both classifiers
            rule_class, rule_conf, rule_scores = self.rule_classifier.classify(features)
            bayes_class, bayes_conf, bayes_probs = self.bayes_classifier.classify(features)
            
            # Use selected classifier for primary result
            if self.classifier_type == 'naive_bayes':
                predicted_class = bayes_class
                confidence = bayes_conf
            else:
                predicted_class = rule_class
                confidence = rule_conf
            
            # Update track's classification
            track.predicted_class = predicted_class
            track.class_confidence = confidence
            track.class_history.append((predicted_class, confidence))
            
            # Keep only last N classifications
            if len(track.class_history) > 10:
                track.class_history = track.class_history[-10:]
            
            classification_results.append({
                'track_id': track.track_id,
                'position': track.pos.tolist(),
                'velocity': track.vel.tolist(),
                'features': features,
                'rule_based': {
                    'class': rule_class,
                    'confidence': rule_conf,
                    'scores': rule_scores
                },
                'naive_bayes': {
                    'class': bayes_class,
                    'confidence': bayes_conf,
                    'probabilities': bayes_probs
                },
                'final_class': predicted_class,
                'final_confidence': confidence,
                'track_age': track.age,
                'track_hits': track.hits
            })
        
        return {
            'timestamp': timestamp,
            'num_detections': len(object_detections),
            'num_clusters': len(clusters),
            'num_noise': len(noise),
            'num_tracks': len(tracks),
            'num_clutter': len(clutter_detections),
            'clusters': clusters,
            'noise': noise,
            'tracks': tracks,
            'classifications': classification_results
        }
    
    def reset(self):
        """Reset pipeline state."""
        self.tracker.reset()
        self.clusterer.next_cluster_id = 1
        self.last_clusters = []
        self.last_noise = []
        self.last_tracks = []
    
    def get_track_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all current tracks."""
        summary = []
        for track in self.last_tracks:
            summary.append({
                'track_id': track.track_id,
                'class': track.predicted_class,
                'confidence': track.class_confidence,
                'position': track.pos.tolist(),
                'speed': float(np.linalg.norm(track.vel)),
                'age': track.age,
                'hits': track.hits,
                'num_points': track.cluster.num_points if track.cluster else 0
            })
        return summary
