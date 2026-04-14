
import cv2
import numpy as np
from time import time
from ..models.yolo import run_yolo, parse_yolo
from ..models.ocr import run_ocr
from .kalman import make_kalman_quad, kalman_update_quad
from ..config import CONF, IOU, SMOOTHING
from .detect import warp_plate, clean_plate

# ════════════════════════════════════════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════════════════════════════════════════

def make_initial_state():
    return {
        "prev_gray":  None,
        "trajectory": [],
        "smoothing":  SMOOTHING,
        "kalman":     None,   # KalmanFilter 16x8
        "missed":     0,
        "max_missed": 10,
    }

# ════════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ════════════════════════════════════════════════════════════════════════════════
def make_pipeline_metrics():
    """Initialise le dictionnaire de métriques pour le pipeline."""
    return {
        "stabilization": [],
        "yolo_inference": [],
        "kalman_postproc": [],
        "ocr": [],
    }

def update_pipeline_metrics(metrics, times):
    """Ajoute les mesures de temps pour une frame."""
    metrics["stabilization"].append(times["stabilization"])
    metrics["yolo_inference"].append(times["yolo_inference"])
    metrics["kalman_postproc"].append(times["kalman_postproc"])
    metrics["ocr"].append(times["ocr"])
    return metrics

def print_pipeline_metrics(metrics):
    """Affiche les statistiques du pipeline en tableau LaTeX."""
    import statistics
    
    data = {}
    total_times = []
    
    for step_name in ["stabilization", "yolo_inference", "kalman_postproc", "ocr"]:
        times = np.array(metrics[step_name]) * 1000  # en ms
        mean = np.mean(times)
        stddev = np.std(times)
        
        data[step_name] = {
            "mean": mean,
            "stddev": stddev,
            "times": times,
        }
        total_times.extend(times)
    
    total_avg = np.mean(total_times)
    
    # ── Affichage LaTeX ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{@{}lrrr@{}}")
    print("\\toprule")
    print("\\textbf{Étape} & \\textbf{Temps moyen} & \\textbf{Écart-type} & \\textbf{\\% total} \\\\")
    print("\\midrule")
    
    for step_name, display_name in [
        ("stabilization", "Stabilisation"),
        ("yolo_inference", "Inférence YOLO"),
        ("kalman_postproc", "Post-traitement"),
        ("ocr", "OCR"),
    ]:
        mean = data[step_name]["mean"]
        stddev = data[step_name]["stddev"]
        percent = 100 * mean / total_avg if total_avg > 0 else 0
        
        print(f"{display_name:20s} & {mean:6.2f} ms & {stddev:6.2f} ms & {percent:5.1f}\\% \\\\")
    
    print("\\midrule")
    print(f"{'TOTAL':20s} & {total_avg:6.2f} ms & -- ms & 100.0\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("="*70 + "\n")
# ════════════════════════════════════════════════════════════════════════════════
# STABILISATION
# ════════════════════════════════════════════════════════════════════════════════

def _get_transform(prev, curr):
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=30)
    if pts is None:
        return np.eye(2, 3, dtype=np.float32)
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None)
    good_prev = pts[status == 1]
    good_curr = curr_pts[status == 1]
    if len(good_prev) < 4:
        return np.eye(2, 3, dtype=np.float32)
    m, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
    return m if m is not None else np.eye(2, 3, dtype=np.float32)


def stabilize(frame, state):
    """(frame, state) → (frame_stabilisée, new_state)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if state["prev_gray"] is None:
        return frame, {**state, "prev_gray": gray, "trajectory": [np.zeros(3)]}

    m        = _get_transform(state["prev_gray"], gray)
    new_traj = state["trajectory"] + [np.array([m[0,2], m[1,2], np.arctan2(m[1,0], m[0,0])])]

    n      = len(new_traj)
    smooth = np.array(new_traj[max(0, n - state["smoothing"]//2):n]).mean(axis=0)
    diff   = smooth - new_traj[-1]

    h, w = frame.shape[:2]
    fix  = np.array([[np.cos(diff[2]), -np.sin(diff[2]), diff[0]],
                     [np.sin(diff[2]),  np.cos(diff[2]), diff[1]]], dtype=np.float32)
    out = cv2.warpAffine(frame, fix, (w, h), borderMode=cv2.BORDER_REFLECT)
    cx, cy = int(w * 0.05), int(h * 0.05)

    return cv2.resize(out[cy:h-cy, cx:w-cx], (w, h)), {**state, "prev_gray": gray, "trajectory": new_traj}


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════════════════════
def run_pipeline(frame, state, yolo, ocr):
    """
    Lance le pipeline complet avec mesure du temps de chaque étape.
    Retourne: (frame, detections, new_state, times_dict)
    """
    times = {}
    
    # ── 1. Stabilisation ─────────────────────────────────────────────────────
    t0 = time()
    frame, new_state = stabilize(frame, state)
    times["stabilization"] = time() - t0

    # ── 2. Détection YOLO ────────────────────────────────────────────────────
    t0 = time()
    outputs    = run_yolo(yolo["sess"], frame, yolo["input_w"], yolo["input_h"], yolo["input_name"])
    detections = parse_yolo(outputs, frame.shape[0], frame.shape[1],
                            yolo["input_h"], yolo["input_w"], yolo["num_masks"])
    times["yolo_inference"] = time() - t0

    # ── 3. Kalman ────────────────────────────────────────────────────────────
    t0 = time()
    if len(detections) > 0:
        poly = detections[0]["polygon"]
        kf   = new_state["kalman"] if new_state["kalman"] is not None \
               else make_kalman_quad(poly)

        kf.predict()
        smoothed_poly = kalman_update_quad(kf, poly)
        detections[0]["polygon"] = smoothed_poly
        new_state = {**new_state, "kalman": kf, "missed": 0}

    else:
        missed = new_state["missed"] + 1
        if new_state["kalman"] is not None and missed <= new_state["max_missed"]:
            new_state["kalman"].predict()
        else:
            new_state = {**new_state, "kalman": None}
        new_state = {**new_state, "missed": missed}
    times["kalman_postproc"] = time() - t0

    # ── 4. OCR ───────────────────────────────────────────────────────────────
    t0 = time()
    for det in detections:
        plate         = warp_plate(frame, det["polygon"]) if len(det["polygon"]) == 4 \
                        else frame[det["box"][1]:det["box"][3], det["box"][0]:det["box"][2]]
        det["text"]   = clean_plate(run_ocr(ocr, plate))
    times["ocr"] = time() - t0

    return frame, detections, new_state, times