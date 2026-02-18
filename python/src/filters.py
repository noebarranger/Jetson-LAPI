from collections import Counter
import numpy as np

def create_kalman_filter(R_val=20.0, Q_val=0.1):
    """Filtre de Kalman pour un état [x, y, w, h, vx, vy, vw, vh]"""
    state = {
        "X": np.zeros((8, 1), dtype=np.float32),
        "P": np.eye(8) * 1.0,
        "initialized": False
    }
    
    F = np.eye(8)
    F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = 1 
    
    H = np.zeros((4, 8))
    H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1
    
    R = np.eye(4) * R_val
    Q = np.eye(8) * Q_val

    def update(measurement):
        if measurement is None:
            if state["initialized"]:
                
                state["X"] = np.dot(F, state["X"])
                state["P"] = np.dot(np.dot(F, state["P"]), F.T) + Q
                return state["X"][:4, 0].tolist()
            return None

        Z = np.array(measurement).reshape(4, 1)
        if not state["initialized"]:
            state["X"][:4, 0] = Z[:, 0]
            state["initialized"] = True

        
        state["X"] = np.dot(F, state["X"])
        state["P"] = np.dot(np.dot(F, state["P"]), F.T) + Q
        
        S = np.dot(np.dot(H, state["P"]), H.T) + R
        K = np.dot(np.dot(state["P"], H.T), np.linalg.inv(S))
        state["X"] = state["X"] + np.dot(K, (Z - np.dot(H, state["X"])))
        state["P"] = state["P"] - np.dot(np.dot(K, H), state["P"])
        return state["X"][:4, 0].tolist()

    return update

def create_stabilizer(window_size=10):
    
    car_kf = create_kalman_filter(R_val=15.0, Q_val=0.2) 
    plate_kf = create_kalman_filter(R_val=10.0, Q_val=0.2)
    history_ocr = []

    def stabilize(current_text, current_kpts, current_car_box):
        
        if current_text:
            history_ocr.append(current_text)
            if len(history_ocr) > window_size: history_ocr.pop(0)
        stable_text = Counter(history_ocr).most_common(1)[0][0] if history_ocr else ""

        
        stable_car_box = None
        if current_car_box:
            x1, y1, x2, y2 = current_car_box
            res = car_kf([x1, y1, x2-x1, y2-y1])
            stable_car_box = [int(res[0]), int(res[1]), int(res[0]+res[2]), int(res[1]+res[3])]
        else:
            car_kf(None)

        
        stable_kpts = None
        if current_kpts:
            pts = np.array(current_kpts)
            center = np.mean(pts, axis=0)
            w_p, h_p = np.max(pts[:,0]) - np.min(pts[:,0]), np.max(pts[:,1]) - np.min(pts[:,1])
            res_p = plate_kf([center[0], center[1], w_p, h_p])
            new_center = np.array([res_p[0], res_p[1]])
            stable_kpts = (new_center + (pts - center)).tolist()
        else:
            plate_kf(None)

        return stable_text, stable_kpts, stable_car_box

    return stabilize