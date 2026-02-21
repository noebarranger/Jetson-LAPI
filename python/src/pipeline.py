# pipeline.py - Version complète fonctionnelle

from typing import List, Callable, Optional
from .datastruct import Detection, TrackSnapshot, PlateDetection
from .geometry import get_warped_plate
import time
import numpy as np


def create_lapi_pipeline(
    run_car: Callable[[np.ndarray], List[Detection]],
    run_plate: Callable[[np.ndarray], Optional[PlateDetection]], 
    run_ocr: Callable[[np.ndarray], str]
):
    
    def process_frame(frame: np.ndarray, current_time: Optional[float] = None) -> List[TrackSnapshot]:
        """
        Pipeline complet: détection -> plaque -> OCR -> snapshots
        """
        if current_time is None:
            current_time = time.time()
        
        results: List[TrackSnapshot] = []
        
        # 1. DÉTECTION VÉHICULES
        detections = run_car(frame)
        print(f"[Pipeline] {len(detections)} véhicules détectés")
        
        # 2. POUR CHAQUE VOITURE: plaque + OCR
        for idx, detection in enumerate(detections):
            x1, y1, x2, y2 = map(int, detection.box)
            h, w = frame.shape[:2]
            
            # Clamp aux limites image
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            car_crop = frame[y1:y2, x1:x2]
            if car_crop.size == 0:
                continue
            
            # Variables plaque
            has_plate = False
            plate_box_abs = None
            plate_keypoints_abs = None
            plate_text = None
            
            # 3. DÉTECTION PLAQUE
            plate_det = run_plate(car_crop)
            
            if plate_det is not None:
                has_plate = True
                
                # Convertir en coordonnées absolues
                plate_abs = plate_det.to_absolute((float(x1), float(y1)))
                
                # Extraire box et keypoints
                plate_box_abs = plate_abs.to_box()
                plate_keypoints_abs = plate_abs.keypoints_rel  # Déjà absolus
                
                # 4. OCR
                try:
                    # Keypoints relatifs au crop pour warping
                    kpts_crop = np.array(plate_det.keypoints_rel, dtype=np.float32)
                    plate_img = get_warped_plate(car_crop, kpts_crop)
                    plate_text = run_ocr(plate_img)                    
                    
                    if plate_text:
                        print(f"[Pipeline] Véhicule {idx}: '{plate_text}'")
                        
                except Exception as e:
                    print(f"[Pipeline] Erreur OCR: {e}")
                    plate_text = None
            
            # 5. CRÉER SNAPSHOT
            v_center = detection.center()
            
            snapshot = TrackSnapshot(
                id=idx + 1,  # ID temporaire
                vehicle_box=detection.box,
                vehicle_center=v_center,
                vehicle_velocity=(0.0, 0.0),  # Pas de tracking
                has_plate=has_plate,
                plate_box=plate_box_abs,
                plate_keypoints=plate_keypoints_abs,
                plate_text=plate_text,
                track_age=1,
                detection_hits=1,
                is_confirmed=True
            )
            
            results.append(snapshot)
        
        print(f"[Pipeline] {len(results)} snapshots créés")
        return results
    
    return process_frame