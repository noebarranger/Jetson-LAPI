from filterpy.kalman import KalmanFilter
import numpy as np

def make_kalman_quad(polygon):
    """
    Filtre de Kalman pour un quadrilatère 4 points.
    State : [x1,y1, x2,y2, x3,y3, x4,y4,  vx1,vy1, vx2,vy2, vx3,vy3, vx4,vy4]
    Mesure : [x1,y1, x2,y2, x3,y3, x4,y4]
    """
    kf = KalmanFilter(dim_x=16, dim_z=8)

    # ── Matrice de transition F : position += vitesse ─────────────────────
    kf.F = np.eye(16)
    for i in range(8):
        kf.F[i, i + 8] = 1.0   # xi += vxi

    # ── Matrice d'observation H : on mesure les 8 coords de position ──────
    kf.H = np.zeros((8, 16))
    for i in range(8):
        kf.H[i, i] = 1.0

    # ── Bruits ────────────────────────────────────────────────────────────
    kf.R *= 2     # bruit mesure  (confiance YOLO)
    kf.P *= 1    # incertitude initiale
    kf.Q *= 0.01    # bruit processus (mouvement régulier)

    # ── Init position ─────────────────────────────────────────────────────
    pts = np.array(polygon, dtype=np.float32).flatten()  # [x1,y1,x2,y2,x3,y3,x4,y4]
    kf.x[:8] = pts.reshape(8, 1)

    return kf


def kalman_predict_quad(kf):
    """Retourne le polygone prédit [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]."""
    kf.predict()
    return kf.x[:8].flatten().reshape(4, 2).astype(int).tolist()


def kalman_update_quad(kf, polygon):
    """Met à jour avec le polygone mesuré — retourne le polygone lissé."""
    pts = np.array(polygon, dtype=np.float32).flatten().reshape(8, 1)
    kf.update(pts)
    return kf.x[:8].flatten().reshape(4, 2).astype(int).tolist()