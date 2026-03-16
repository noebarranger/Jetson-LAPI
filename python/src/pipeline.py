# pipeline.py - Version avec tracking Kalman

from typing import List, Callable, Optional, Dict
from .datastruct import Detection, TrackSnapshot, PlateDetection, TrackInternal, SystemState, KalmanState
from .geometry import get_warped_plate
from .filters import (
    create_kalman_vehicle, create_kalman_plate,
    kalman_predict_vehicle, kalman_update_vehicle,
    kalman_predict_plate, kalman_update_plate,
    kalman_to_box_vehicle, kalman_to_box_plate,
    box_to_measurement_plate, compute_iou, compute_distance_cost
)
import time
import numpy as np


# Paramètres du tracker
MAX_AGE = 30          # Frames avant suppression
MIN_HITS = 5          # Hits minimum pour confirmation
IOU_THRESHOLD = 0.3   # Seuil d'association
DISTANCE_THRESHOLD = 1.0  # Seuil de distance


def create_lapi_pipeline(
    run_car: Callable[[np.ndarray], List[Detection]],
    run_plate: Callable[[np.ndarray], Optional[PlateDetection]], 
    run_ocr: Callable[[np.ndarray], str]
):
    
    # État global du système (mutable via nonlocal)
    system_state = SystemState(
        tracks={},
        next_id=1,
        frame_count=0
    )
    
    def process_frame(frame: np.ndarray, current_time: Optional[float] = None) -> List[TrackSnapshot]:
        """
        Pipeline complet avec tracking Kalman.
        """
        nonlocal system_state
        
        if current_time is None:
            current_time = time.time()
        
        system_state = SystemState(
            tracks=system_state.tracks,
            next_id=system_state.next_id,
            frame_count=system_state.frame_count + 1
        )
        
        # 1. DÉTECTION VÉHICULES
        detections = run_car(frame)
        print(f"[Pipeline] Frame {system_state.frame_count}: {len(detections)} détections")
        
        # 2. PRÉDICTION KALMAN pour tous les tracks existants
        predicted_tracks = {}
        for track_id, track in system_state.tracks.items():
            # Prédire véhicule
            new_kalman = kalman_predict_vehicle(track.kalman)
            
            # Prédire plaque si existe
            new_plate_kalman = None
            if track.plate_kalman is not None:
                new_plate_kalman = kalman_predict_plate(track.plate_kalman)
            
            predicted_tracks[track_id] = TrackInternal(
                track_id=track.track_id,
                kalman=new_kalman,
                hits=track.hits,
                misses=track.misses,
                age=track.age + 1,
                birth_frame=track.birth_frame,
                plate_keypoints=track.plate_keypoints,
                plate_kalman=new_plate_kalman,
                plate_hits=track.plate_hits,
                plate_misses=track.plate_misses,
                ocr_history=track.ocr_history
            )
        
        # 3. ASSOCIATION DÉTECTIONS ↔ TRACKS (Hungarian algorithm simplifié)
        matches, unmatched_detections, unmatched_tracks = associate_detections_to_tracks(
            detections, predicted_tracks, frame.shape[1], frame.shape[0]
        )
        
        # 4. MISE À JOUR DES TRACKS ASSOCIÉS
        updated_tracks = {}
        
        for track_id, det_idx in matches:
            track = predicted_tracks[track_id]
            detection = detections[det_idx]
            
            # Mise à jour Kalman véhicule
            measurement = detection.to_xywh()  # [cx, cy, w, h]
            new_kalman = kalman_update_vehicle(track.kalman, measurement)
            
            # Traitement plaque
            has_plate, plate_keypoints, plate_kalman, plate_text, new_ocr_history = update_plate(
                track, detection, frame, run_plate, run_ocr
            )
            
            updated_tracks[track_id] = TrackInternal(
                track_id=track.track_id,
                kalman=new_kalman,
                hits=track.hits + 1,
                misses=0,
                age=track.age,
                birth_frame=track.birth_frame,
                plate_keypoints=plate_keypoints,
                plate_kalman=plate_kalman,
                plate_hits=track.plate_hits + (1 if has_plate else 0),
                plate_misses=track.plate_misses + (0 if has_plate else 1),
                ocr_history=new_ocr_history
            )
        
        # 5. GÉRER LES TRACKS NON ASSOCIÉS (misses)
        for track_id in unmatched_tracks:
            track = predicted_tracks[track_id]
            
            if track.misses + 1 > MAX_AGE:
                continue  # Supprimer (ne pas ajouter à updated_tracks)
            
            updated_tracks[track_id] = TrackInternal(
                track_id=track.track_id,
                kalman=track.kalman,
                hits=track.hits,
                misses=track.misses + 1,
                age=track.age,
                birth_frame=track.birth_frame,
                plate_keypoints=track.plate_keypoints,
                plate_kalman=track.plate_kalman,
                plate_hits=track.plate_hits,
                plate_misses=track.plate_misses + 1,
                ocr_history=track.ocr_history
            )
        
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            
            # Nouveau track avec Kalman initialisé
            kalman = create_kalman_vehicle(detection.box)
            
            # Traitement plaque initiale
            has_plate, plate_keypoints, plate_kalman, plate_text, ocr_history = update_plate(
                None, detection, frame, run_plate, run_ocr
            )
            
            new_track = TrackInternal(
                track_id=system_state.next_id,
                kalman=kalman,
                hits=1,
                misses=0,
                age=1,
                birth_frame=system_state.frame_count,
                plate_keypoints=plate_keypoints,
                plate_kalman=plate_kalman,
                plate_hits=1 if has_plate else 0,
                plate_misses=0 if has_plate else 1,
                ocr_history=ocr_history
            )
            
            updated_tracks[system_state.next_id] = new_track
            system_state = SystemState(
                tracks=updated_tracks,
                next_id=system_state.next_id + 1,
                frame_count=system_state.frame_count
            )
        
        # 7. METTRE À JOUR L'ÉTAT GLOBAL
        system_state = SystemState(
            tracks=updated_tracks,
            next_id=system_state.next_id,
            frame_count=system_state.frame_count
        )
        
        # 8. GÉNÉRER LES SNAPSHOTS DE SORTIE
        snapshots = []
        for track_id, track in updated_tracks.items():
            # Ne renvoyer que les tracks confirmées
            if track.hits < MIN_HITS and track.age < MIN_HITS * 2:
                continue
            
            # Calculer vélocité depuis Kalman
            vx, vy = track.kalman.mean[4], track.kalman.mean[5]
            
            # Box prédite/filtrée par Kalman
            v_box = kalman_to_box_vehicle(track.kalman)
            v_center = (track.kalman.mean[0], track.kalman.mean[1])
            
            # Box plaque absolue
            p_box = None
            if track.plate_kalman is not None:
                p_box = kalman_to_box_plate(track.plate_kalman, v_box)
            
            # Texte OCR
            plate_text = None
            if track.ocr_history:
                # Prendre le texte le plus fréquent
                from collections import Counter
                text_counts = Counter(track.ocr_history)
                plate_text = text_counts.most_common(1)[0][0]
            
            snapshot = TrackSnapshot(
                id=track.track_id,
                vehicle_box=v_box,
                vehicle_center=v_center,
                vehicle_velocity=(vx, vy),
                has_plate=track.plate_keypoints is not None,
                plate_box=p_box,
                plate_keypoints=track.plate_keypoints,
                plate_text=plate_text,
                track_age=track.age,
                detection_hits=track.hits,
                is_confirmed=track.hits >= MIN_HITS
            )
            snapshots.append(snapshot)
        
        print(f"[Pipeline] {len(snapshots)} tracks actifs")
        return snapshots
    
    return process_frame


def associate_detections_to_tracks(
    detections: List[Detection],
    tracks: Dict[int, TrackInternal],
    img_w: int,
    img_h: int
) -> tuple:
    """
    Association simple basée sur IOU et distance.
    Retourne: (matches, unmatched_detections, unmatched_tracks)
    """
    if not tracks:
        return [], list(range(len(detections))), []
    
    if not detections:
        return [], [], list(tracks.keys())
    
    # Calculer la matrice de coût
    track_ids = list(tracks.keys())
    cost_matrix = np.zeros((len(detections), len(track_ids)))
    
    for i, det in enumerate(detections):
        for j, track_id in enumerate(track_ids):
            track = tracks[track_id]
            pred_box = kalman_to_box_vehicle(track.kalman)
            
            # Combinaison IOU + distance
            iou = compute_iou(det.box, pred_box)
            dist_cost = compute_distance_cost(track.kalman, det)
            
            # Coût combiné (plus c'est bas, mieux c'est)
            if iou > IOU_THRESHOLD and dist_cost < DISTANCE_THRESHOLD:
                cost_matrix[i, j] = 1.0 - iou + dist_cost * 0.1
            else:
                cost_matrix[i, j] = 1e6  # Invalide
    
    # Association greedy simple (peut être remplacé par Hungarian)
    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_tracks = list(track_ids)
    
    # Trier par coût croissant
    indices = np.argsort(cost_matrix.flatten())
    
    used_dets = set()
    used_tracks = set()
    
    for idx in indices:
        if cost_matrix.flatten()[idx] >= 1e6:
            break
        
        i = idx // len(track_ids)
        j = idx % len(track_ids)
        track_id = track_ids[j]
        
        if i in used_dets or track_id in used_tracks:
            continue
        
        matches.append((track_id, i))
        used_dets.add(i)
        used_tracks.add(track_id)
        
        if i in unmatched_detections:
            unmatched_detections.remove(i)
        if track_id in unmatched_tracks:
            unmatched_tracks.remove(track_id)
    
    return matches, unmatched_detections, unmatched_tracks


def update_plate(track, detection, frame, run_plate, run_ocr):
    """
    Met à jour la détection de plaque pour un track.
    Retourne: (has_plate, keypoints, kalman_state, text, ocr_history)
    """
    x1, y1, x2, y2 = map(int, detection.box)
    h, w = frame.shape[:2]
    
    # Clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        # Conserver l'ancienne plaque si existe
        if track is not None:
            return (track.plate_keypoints is not None, track.plate_keypoints,
                   track.plate_kalman, None, track.ocr_history)
        return False, None, None, None, ()
    
    car_crop = frame[y1:y2, x1:x2]
    if car_crop.size == 0:
        if track is not None:
            return (track.plate_keypoints is not None, track.plate_keypoints,
                   track.plate_kalman, None, track.ocr_history)
        return False, None, None, None, ()
    
    # Détection plaque
    plate_det = run_plate(car_crop)
    
    if plate_det is None:
        # Pas de détection, conserver l'ancienne si récente
        if track is not None and track.plate_misses < 5:
            return (track.plate_keypoints is not None, track.plate_keypoints,
                   track.plate_kalman, None, track.ocr_history)
        return False, None, None, None, () if track is None else track.ocr_history
    
    # Convertir en coordonnées absolues
    plate_abs = plate_det.to_absolute((float(x1), float(y1)))
    plate_box = plate_abs.to_box()
    plate_keypoints = plate_abs.keypoints_rel
    
    # Initialiser ou mettre à jour Kalman plaque
    if track is None or track.plate_kalman is None:
        plate_kalman = create_kalman_plate(plate_box, detection.box)
    else:
        measurement = box_to_measurement_plate(plate_box, detection.box)
        plate_kalman = kalman_update_plate(track.plate_kalman, measurement)
    
    # OCR
    plate_text = None
    try:
        kpts_crop = np.array(plate_det.keypoints_rel, dtype=np.float32)
        plate_img = get_warped_plate(car_crop, kpts_crop)
        plate_text = run_ocr(plate_img)
    except Exception as e:
        print(f"[Pipeline] Erreur OCR: {e}")
    
    # Mettre à jour historique OCR
    ocr_history = () if track is None else track.ocr_history
    if plate_text:
        ocr_history = ocr_history + (plate_text,)
        # Garder les 10 derniers
        if len(ocr_history) > 10:
            ocr_history = ocr_history[-10:]
    
    return True, plate_keypoints, plate_kalman, plate_text, ocr_history