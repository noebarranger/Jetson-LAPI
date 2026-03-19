
import cv2
import numpy as np
from ..models.yolo import run_yolo, parse_yolo
from ..config import CONF, IOU, SMOOTHING


# ════════════════════════════════════════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════════════════════════════════════════

def make_initial_state():
    return {
        "prev_gray":  None,
        "trajectory": [],
        "smoothing":  SMOOTHING,
    }


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

def run_pipeline(frame, state, yolo):
    """
    (frame, state, yolo) → (frame_stabilisée, détections, new_state)
    """
    frame, new_state = stabilize(frame, state)

    outputs    = run_yolo(yolo["sess"], frame, yolo["input_w"], yolo["input_h"], yolo["input_name"])
    detections = parse_yolo(outputs, frame.shape[0], frame.shape[1], yolo["input_h"], yolo["input_w"], yolo["num_masks"])

    return frame, detections, new_state