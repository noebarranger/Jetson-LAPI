import numpy as np
import cv2
import re 

def order_points(pts):
    pts  = np.array(pts, dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)


def warp_plate(frame, polygon, out_w=200, out_h=60):
    if len(polygon) != 4:
        return None
    src = order_points(polygon)
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst),
                               (out_w, out_h), flags=cv2.INTER_CUBIC)

def clean_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9\s]', '', text)

    for pattern in [r'[A-Z]{3}\s?\d{3}', r'\d{3}\s?[A-Z]{3}', r'[A-Z]\d{2}\s?[A-Z]{3}']:
        m = re.search(pattern, text)
        if m:
            return m.group(0)
    return text