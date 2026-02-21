import numpy as np
from typing import Tuple, List, Dict, Optional
from .datastruct import (
    Detection, 
    TrackInternal, 
    SystemState,
    TrackSnapshot,
    KalmanState,
    PlateDetection
)

def compute_iou(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    """Pure IOU calculation"""
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    
    xi1, yi1 = max(x1_a, x1_b), max(y1_a, y1_b)
    xi2, yi2 = min(x2_a, x2_b), min(y2_a, y2_b)
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h
    
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union = area_a + area_b - inter
    
    return inter / union if union > 0 else 0.0

# ============================================================
# KALMAN POUR VÉHICULE (8 états: x, y, w, h, vx, vy, vw, vh)
# ============================================================

def create_kalman_vehicle(initial_box: Tuple[float, float, float, float], 
                         dt: float = 1.0) -> KalmanState:
    """
    Initialise un filtre de Kalman 8D pour un véhicule.
    États: [cx, cy, w, h, vcx, vcy, vw, vh]
    """
    x1, y1, x2, y2 = initial_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    
    # État initial [cx, cy, w, h, vcx, vcy, vw, vh]
    mean = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Covariance initiale (incertitude élevée sur vitesses)
    covariance = np.diag([
        w * 0.1, h * 0.1, w * 0.1, h * 0.1,  # position/taille
        100.0, 100.0, 100.0, 100.0              # vitesses (inconnues)
    ])
    
    # Matrice de transition F (modèle constant velocity)
    F = np.array([
        [1, 0, 0, 0, dt, 0, 0, 0],
        [0, 1, 0, 0, 0, dt, 0, 0],
        [0, 0, 1, 0, 0, 0, dt, 0],
        [0, 0, 0, 1, 0, 0, 0, dt],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=np.float32)
    
    # Matrice d'observation H (on observe que cx, cy, w, h)
    H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
    ], dtype=np.float32)
    
    # Bruit de process Q (modèle non parfait)
    Q = np.diag([
        0.1, 0.1, 0.1, 0.1,      # position
        1.0, 1.0, 0.5, 0.5       # vitesses (plus de bruit)
    ])
    
    # Bruit de mesure R (détection YOLO)
    R = np.diag([
        w * 0.05, h * 0.05, w * 0.05, h * 0.05  # 5% d'incertitude
    ])
    
    return KalmanState(
        mean=tuple(mean),
        covariance_diag=tuple(np.diag(covariance)),
        F=tuple(map(tuple, F)),
        H=tuple(map(tuple, H)),
        Q_diag=tuple(np.diag(Q)),
        R_diag=tuple(np.diag(R))
    )


def kalman_predict_vehicle(state: KalmanState, dt: float = 1.0) -> KalmanState:
    """Prédiction Kalman pour véhicule"""
    mean = np.array(state.mean, dtype=np.float32)
    covariance = np.diag(state.covariance_diag)
    F = np.array(state.F, dtype=np.float32)
    Q = np.diag(state.Q_diag)
    
    # Prédiction
    mean_pred = F @ mean
    cov_pred = F @ covariance @ F.T + Q
    
    return KalmanState(
        mean=tuple(mean_pred),
        covariance_diag=tuple(np.diag(cov_pred)),
        F=state.F,
        H=state.H,
        Q_diag=state.Q_diag,
        R_diag=state.R_diag
    )


def kalman_update_vehicle(state: KalmanState, 
                         measurement: Tuple[float, float, float, float]) -> KalmanState:
    """Mise à jour Kalman avec nouvelle détection"""
    mean = np.array(state.mean, dtype=np.float32)
    covariance = np.diag(state.covariance_diag)
    H = np.array(state.H, dtype=np.float32)
    R = np.diag(state.R_diag)
    
    z = np.array(measurement, dtype=np.float32)
    
    # Innovation
    y = z - H @ mean
    S = H @ covariance @ H.T + R
    
    # Gain de Kalman
    K = covariance @ H.T @ np.linalg.inv(S)
    
    # Mise à jour
    mean_new = mean + K @ y
    cov_new = (np.eye(8) - K @ H) @ covariance
    
    return KalmanState(
        mean=tuple(mean_new),
        covariance_diag=tuple(np.diag(cov_new)),
        F=state.F,
        H=state.H,
        Q_diag=state.Q_diag,
        R_diag=state.R_diag
    )


# ============================================================
# KALMAN POUR PLAQUE (4 états: dx, dy, pw, ph - relatif au véhicule)
# ============================================================

def create_kalman_plate(initial_box: Tuple[float, float, float, float],
                       vehicle_box: Tuple[float, float, float, float]) -> KalmanState:
    """
    Initialise Kalman 4D pour plaque (position relative au véhicule).
    États: [dx, dy, pw, ph] où dx,dy = offset du centre plaque par rapport au centre véhicule
    """
    vx1, vy1, vx2, vy2 = vehicle_box
    v_cx, v_cy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
    
    px1, py1, px2, py2 = initial_box
    p_cx, p_cy = (px1 + px2) / 2, (py1 + py2) / 2
    pw, ph = px2 - px1, py2 - py1
    
    # Position relative
    dx, dy = p_cx - v_cx, p_cy - v_cy
    
    mean = np.array([dx, dy, pw, ph], dtype=np.float32)
    
    covariance = np.diag([
        (vx2 - vx1) * 0.1, (vy2 - vy1) * 0.1,  # position relative
        pw * 0.1, ph * 0.1                      # taille
    ])
    
    # Modèle statique (la plaque bouge avec le véhicule)
    F = np.eye(4, dtype=np.float32)
    
    # Observation directe
    H = np.eye(4, dtype=np.float32)
    
    # Bruits
    Q = np.diag([5.0, 5.0, 2.0, 2.0])  # Bruit process
    R = np.diag([10.0, 10.0, 5.0, 5.0])  # Bruit mesure
    
    return KalmanState(
        mean=tuple(mean),
        covariance_diag=tuple(np.diag(covariance)),
        F=tuple(map(tuple, F)),
        H=tuple(map(tuple, H)),
        Q_diag=tuple(np.diag(Q)),
        R_diag=tuple(np.diag(R))
    )


def kalman_predict_plate(state: KalmanState) -> KalmanState:
    """Prédiction pour plaque (modèle statique)"""
    mean = np.array(state.mean, dtype=np.float32)
    covariance = np.diag(state.covariance_diag)
    F = np.array(state.F, dtype=np.float32)
    Q = np.diag(state.Q_diag)
    
    mean_pred = F @ mean
    cov_pred = F @ covariance @ F.T + Q
    
    return KalmanState(
        mean=tuple(mean_pred),
        covariance_diag=tuple(np.diag(cov_pred)),
        F=state.F,
        H=state.H,
        Q_diag=state.Q_diag,
        R_diag=state.R_diag
    )


def kalman_update_plate(state: KalmanState,
                       measurement: Tuple[float, float, float, float]) -> KalmanState:
    """Mise à jour plaque"""
    mean = np.array(state.mean, dtype=np.float32)
    covariance = np.diag(state.covariance_diag)
    H = np.array(state.H, dtype=np.float32)
    R = np.diag(state.R_diag)
    
    z = np.array(measurement, dtype=np.float32)
    
    y = z - H @ mean
    S = H @ covariance @ H.T + R
    K = covariance @ H.T @ np.linalg.inv(S)
    
    mean_new = mean + K @ y
    cov_new = (np.eye(4) - K @ H) @ covariance
    
    return KalmanState(
        mean=tuple(mean_new),
        covariance_diag=tuple(np.diag(cov_new)),
        F=state.F,
        H=state.H,
        Q_diag=state.Q_diag,
        R_diag=state.R_diag
    )


# ============================================================
# UTILITAIRES DE CONVERSION
# ============================================================

def kalman_to_box_vehicle(state: KalmanState) -> Tuple[float, float, float, float]:
    """Convertit état Kalman véhicule en box [x1,y1,x2,y2]"""
    cx, cy, w, h = state.mean[:4]
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)


def kalman_to_box_plate(state: KalmanState, 
                       vehicle_box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Convertit état Kalman plaque (relatif) en box absolue"""
    dx, dy, pw, ph = state.mean
    v_cx = (vehicle_box[0] + vehicle_box[2]) / 2
    v_cy = (vehicle_box[1] + vehicle_box[3]) / 2
    
    p_cx, p_cy = v_cx + dx, v_cy + dy
    return (p_cx - pw/2, p_cy - ph/2, p_cx + pw/2, p_cy + ph/2)


def box_to_measurement_plate(plate_box: Tuple[float, float, float, float],
                            vehicle_box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Convertit box absolue en mesure relative pour Kalman plaque"""
    px1, py1, px2, py2 = plate_box
    p_cx, p_cy = (px1 + px2) / 2, (py1 + py2) / 2
    pw, ph = px2 - px1, py2 - py1
    
    v_cx = (vehicle_box[0] + vehicle_box[2]) / 2
    v_cy = (vehicle_box[1] + vehicle_box[3]) / 2
    
    return (p_cx - v_cx, p_cy - v_cy, pw, ph)

def compute_distance_cost(track_state: KalmanState, 
                         detection: Detection) -> float:
    """
    Cost basé sur la distance Mahalanobis entre prédiction et détection.
    Plus précis que IOU pour les mouvements rapides.
    """
    pred_box = kalman_to_box_vehicle(track_state)
    pred_center = ((pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2)
    det_center = detection.center()
    
    # Distance euclidienne simple (peut être améliorée avec Mahalanobis complète)
    dist = np.sqrt((pred_center[0] - det_center[0])**2 + 
                   (pred_center[1] - det_center[1])**2)
    
    # Normaliser par la taille de la box prédite
    w, h = pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]
    norm_dist = dist / np.sqrt(w * h)
    
    return norm_dist