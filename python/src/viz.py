import cv2
import numpy as np

from typing import Tuple, Optional, List
from .datastruct import TrackSnapshot


def draw_tracks(frame, tracks: List[TrackSnapshot]) -> np.ndarray:
    result = frame.copy()
    [draw_track(result, track) for track in tracks]
    return result 

def draw_track(frame, track: TrackSnapshot) -> np.ndarray:
    """
    Dessine une track avec informations de tracking Kalman.
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    color = colors[track.id % len(colors)]
    
    # --- VÉHICULE ---
    x1, y1, x2, y2 = map(int, track.vehicle_box)
    
    # Filtre taille minimale
    MIN_WIDTH, MIN_HEIGHT = 60, 30  # Réduit pour voir les lointains
    w, h = x2 - x1, y2 - y1
    
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return frame
    
    # Box véhicule (épaisseur selon confiance)
    thickness = 3 if track.is_confirmed else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Label avec ID, vélocité et âge
    vel_x, vel_y = track.vehicle_velocity
    speed = np.sqrt(vel_x**2 + vel_y**2)
    status = "CONF" if track.is_confirmed else "TENT"
    label = f"ID:{track.id} {status} V:{speed:.1f} A:{track.track_age}"
    
    # Fond pour le label
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Vecteur vitesse (prédiction Kalman)
    if abs(vel_x) > 1 or abs(vel_y) > 1:
        cx, cy = int(track.vehicle_center[0]), int(track.vehicle_center[1])
        cv2.arrowedLine(frame, (cx, cy), 
                       (int(cx + vel_x * 5), int(cy + vel_y * 5)), 
                       (0, 255, 255), 2)
    
    # --- PLAQUE ---
    if track.has_plate and track.plate_box:
        px1, py1, px2, py2 = map(int, track.plate_box)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
        
        if track.plate_text:
            text = f"{track.plate_text} ({track.detection_hits} hits)"
            y_text = py1 - 10 if py1 > 30 else py2 + 20
            
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, 
                         (px1, y_text - th - 5), 
                         (px1 + tw, y_text + 5), 
                         (0, 0, 0), -1)
            cv2.putText(frame, text, (px1, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if track.plate_keypoints:
            pts = np.array(track.plate_keypoints, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    
    return frame