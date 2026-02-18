import cv2
import numpy as np

def draw_hud(frame, car_box, kpts, text, conf=None):
    res_img = frame.copy()
    
    # Dessin Voiture stabilisée
    if car_box:
        x1, y1, x2, y2 = map(int, car_box)
        cv2.rectangle(res_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(res_img, "VEHICLE", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Dessin Plaque stabilisée
    if kpts:
        pts_poly = np.array(kpts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(res_img, [pts_poly], isClosed=True, color=(0, 255, 0), thickness=2)
        
        if text:
            # Ancrage dynamique du texte au-dessus de la plaque
            x_t, y_t = int(kpts[0][0]), int(kpts[0][1]) - 15
            cv2.putText(res_img, f"PLATE: {text}", (x_t, y_t), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    return res_img