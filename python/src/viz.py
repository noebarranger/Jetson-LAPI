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
    Dessine une seule track (véhicule + plaque si présente).
    """
    # Couleur unique par ID
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    color = colors[track.id % len(colors)]
    
    # --- VÉHICULE ---
    x1, y1, x2, y2 = map(int, track.vehicle_box)
    
    # Filtre taille minimale
    MIN_WIDTH, MIN_HEIGHT = 100, 48
    w, h = x2 - x1, y2 - y1
    
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return frame
    
    # Box véhicule
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Label véhicule
    vel_x, vel_y = track.vehicle_velocity
    speed = np.sqrt(vel_x**2 + vel_y**2)
    label = f"ID:{track.id} V:{speed:.1f}px/f"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # --- PLAQUE ---
    if track.has_plate and track.plate_box:
        px1, py1, px2, py2 = map(int, track.plate_box)
        
        # Box plaque
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
        
        # Texte OCR
        if track.plate_text:
            conf_str = ""
            if hasattr(track, 'plate_confidence') and track.plate_confidence is not None:
                conf_str = f" ({track.plate_confidence:.2f})"
            
            text = f"{track.plate_text}{conf_str}"
            
            y_text = py1 - 10 if py1 > 30 else py2 + 20
            
            # Fond noir pour lisibilité
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, 
                         (px1, y_text - th - 5), 
                         (px1 + tw, y_text + 5), 
                         (0, 0, 0), -1)
            
            # Texte en vert
            cv2.putText(frame, text, 
                       (px1, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if track.plate_keypoints:
            pts = np.array(track.plate_keypoints, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    
    return frame