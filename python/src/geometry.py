import cv2
import numpy as np

def get_warped_plate(img, kpts):
    src_pts = order_points(kpts)
    
    # Ratio plaque standard européenne ~4:1 ou 5:1
    # Pour utiliser les 320px de largeur PP-OCRv4 : 320/48 = 6.66 ratio max
    
    width, height = 480, 96
    
    dst_pts = np.array([[0, 0], 
                        [width-1, 0], 
                        [width-1, height-1], 
                        [0, height-1]], 
                        dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (width, height))


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Haut-Gauche
    rect[2] = pts[np.argmax(s)] # Bas-Droite
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Haut-Droite
    rect[3] = pts[np.argmax(diff)] # Bas-Gauche
    
    return rect