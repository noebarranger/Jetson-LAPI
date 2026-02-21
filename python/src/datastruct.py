# datastruct.py - Correction détection plaque 4 points

from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import numpy as np


# ============================================================
# DONNÉES ENTRANTES (Immutable)
# ============================================================

@dataclass(frozen=True)
class Detection:
    """Détection brute YOLO voiture"""
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0
    
    def center(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2,
                (self.box[1] + self.box[3]) / 2)
    
    def to_xywh(self) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        return (cx, cy, w, h)


@dataclass(frozen=True)
class PlateDetection:
    """
    Détection YOLO-pose: 4 points en coordonnées RELATIVES au crop voiture
    Ordre: Haut-Gauche, Haut-Droite, Bas-Droite, Bas-Gauche (ou selon order_points)
    """
    # 4 points = 8 coordonnées: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    keypoints_rel: Tuple[
        Tuple[float, float],  # P1: HG
        Tuple[float, float],  # P2: HD  
        Tuple[float, float],  # P3: BD
        Tuple[float, float]   # P4: BG
    ]
    confidence: float  # Confiance moyenne des keypoints
    
    def to_absolute(self, crop_offset: Tuple[float, float]) -> 'PlateDetection':
        """Convertit en coordonnées image absolues"""
        dx, dy = crop_offset
        abs_pts = tuple((p[0] + dx, p[1] + dy) for p in self.keypoints_rel)
        return PlateDetection(keypoints_rel=abs_pts, confidence=self.confidence)
    
    def to_box(self) -> Tuple[float, float, float, float]:
        """Convertit 4 points en bounding box [x1,y1,x2,y2]"""
        xs = [p[0] for p in self.keypoints_rel]
        ys = [p[1] for p in self.keypoints_rel]
        return (min(xs), min(ys), max(xs), max(ys))
    
    def center(self) -> Tuple[float, float]:
        """Centre de la plaque (moyenne des 4 points)"""
        box = self.to_box()
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


# ============================================================
# SORTIE (Immutable)
# ============================================================

@dataclass(frozen=True)
class TrackSnapshot:
    """Sortie immutable"""
    id: int
    vehicle_box: Tuple[float, float, float, float]
    vehicle_center: Tuple[float, float]
    vehicle_velocity: Tuple[float, float]
    
    # Plaque
    has_plate: bool
    plate_box: Optional[Tuple[float, float, float, float]]  # Depuis keypoints
    plate_keypoints: Optional[Tuple[
        Tuple[float, float],
        Tuple[float, float], 
        Tuple[float, float],
        Tuple[float, float]
    ]]  # Les 4 points originaux (pour warping OCR)
    plate_text: Optional[str]
    
    track_age: int
    detection_hits: int
    is_confirmed: bool


# ============================================================
# ÉTAT INTERNE (Immutable)
# ============================================================

@dataclass(frozen=True)
class KalmanState:
    """État Kalman"""
    mean: Tuple[float, ...]           # (8,) pour véhicule, (4,) pour plaque
    covariance_diag: Tuple[float, ...]
    F: Tuple[Tuple[float, ...], ...]
    H: Tuple[Tuple[float, ...], ...]
    Q_diag: Tuple[float, ...]
    R_diag: Tuple[float, ...]


@dataclass(frozen=True)
class TrackInternal:
    """État interne track"""
    track_id: int
    kalman: KalmanState  # Véhicule 8D
    
    hits: int
    misses: int
    age: int
    birth_frame: int
    
    # Plaque: stocke les 4 points + Kalman 4D relatif
    plate_keypoints: Optional[Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float]
    ]] = None
    
    plate_kalman: Optional[KalmanState] = None  # [dx, dy, pw, ph]
    plate_hits: int = 0
    plate_misses: int = 0
    ocr_history: Tuple[str, ...] = ()

@dataclass(frozen=True)
class SystemState:
    """État global"""
    tracks: Dict[int, TrackInternal]
    next_id: int
    frame_count: int