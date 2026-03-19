import math
import re
import cv2
import numpy as np
import onnxruntime as ort



def load_ocr(path, chars_path, providers):
    sess = ort.InferenceSession(path, providers=providers)
    with open(chars_path, 'r', encoding='utf-8') as f:
        chars = [''] + [c.strip() for c in f.readlines() if c.strip()]
    print(f"OCR  : {sess.get_providers()}, {len(chars)} chars")
    return {
        "sess":  sess,
        "chars": chars,
    }


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

def apply_clahe(img):
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab     = cv2.merge((cv2.createCLAHE(3.0, (8,8)).apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def decode_ocr(preds, chars):
    idx  = preds.argmax(axis=2)[0]
    text = ""
    for i in range(len(idx)):
        c = idx[i]
        if c != 0 and (i == 0 or c != idx[i-1]) and c < len(chars):
            text += chars[c]
    return text


def run_ocr(ocr_sess, plate_img):
    if plate_img is None or plate_img.size == 0:
        return ""
    imgC, imgH, imgW = 3, 48, 320
    h, w      = plate_img.shape[:2]
    resized_w = min(int(imgH * w / float(h)), imgW)

    variants = [
        plate_img,
        cv2.bitwise_not(plate_img),
        cv2.filter2D(plate_img, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])),
    ]

    results = []
    for v in variants:
        img    = cv2.cvtColor(apply_clahe(v), cv2.COLOR_BGR2RGB)
        img    = cv2.resize(img, (resized_w, imgH))
        img    = (img.astype('float32') / 255.0 - 0.5) / 0.5
        tensor = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        tensor[:, :, 0:resized_w] = img.transpose((2, 0, 1))
        preds  = ocr_sess.run(None, {ocr_sess.get_inputs()[0].name: tensor[np.newaxis]})[0]
        results.append(decode_ocr(preds))

    return max(results, key=len)

def clean_plate(text):
    text  = text.upper()

    patterns = [
        r'[A-Z]{3}\s?\d{3}',    # ABC 123
        r'\d{3}\s?[A-Z]{3}',    # 123 ABC
        r'[A-Z]\d{2}\s?[A-Z]{3}',  # A12 BCD
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return text
