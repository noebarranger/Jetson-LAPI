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
def apply_mask(text, mask):
    """
    text : "Y4555T"   (texte brut sans espaces)
    mask : "LDDLLL"   (L=lettre, D=chiffre)
    Retourne "Y4___T" — _ aux positions où le char ne correspond pas au type attendu
    """
    result = ""
    for char, expected in zip(text, mask):
        if expected == 'L' and char.isalpha():
            result += char
        elif expected == 'D' and char.isdigit():
            result += char
        else:
            result += '_'
    return result

def clean_plate(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())

    patterns = [
        (r'[A-Z0-9]{6}', 'LLLDDD'),   # ABC123
        (r'[A-Z0-9]{6}', 'DDDLLL'),   # 123ABC
        (r'[A-Z0-9]{6}', 'LDDLLL'),   # A12BCD
        (r'[A-Z0-9]{6}', 'LLLDDL'),   # AAK70N
        (r'[A-Z0-9]{6}', 'DDLLLL'),   # 23AAGP
    ]

    if len(text) < 6:
        return ""

    best = ""
    best_score = -1

    for regex, mask in patterns:
        m = re.search(regex, text)
        if not m:
            continue
        candidate = m.group(0)
        masked    = apply_mask(candidate, mask)
        score     = sum(1 for c in masked if c != '_')  # nb de chars valides

        if score > best_score:
            best_score = score
            best       = masked

    return best   # ex: "Y4___T" ou "Y45SST"
