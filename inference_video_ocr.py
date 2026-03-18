#!/usr/bin/env python3
import math
import re
import cv2
import numpy as np
import onnxruntime as ort

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = '/app/runs/train_merged6/weights/best.onnx'
OCR_MODEL    = '/app/ocr/ocr_model.onnx'
CHARS_PATH   = '/app/ocr/en_dict.txt'
INPUT_VIDEO  = '/app/segment_001.avi'
OUTPUT_VIDEO = '/app/output/resultat_ocr3.mp4'
CONF      = 0.5
IOU       = 0.5
SMOOTHING = 100
LABELS    = ['day', 'night']
COLORS    = [(0, 255, 0), (0, 0, 255)]

# ── Modèles ───────────────────────────────────────────────────────────────────
yolo_sess  = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = yolo_sess.get_inputs()[0].name
input_h    = yolo_sess.get_inputs()[0].shape[2]
input_w    = yolo_sess.get_inputs()[0].shape[3]
num_masks  = 32
print(f"YOLO  : {yolo_sess.get_providers()}")

ocr_sess = ort.InferenceSession(OCR_MODEL, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
with open(CHARS_PATH, 'r', encoding='utf-8') as f:
    chars = [''] + [c.strip() for c in f.readlines() if c.strip()]
print(f"OCR   : {ocr_sess.get_providers()}, {len(chars)} chars")


# ════════════════════════════════════════════════════════════════════════════════
# STABILISATION
# ════════════════════════════════════════════════════════════════════════════════
class VideoStabilizer:
    def __init__(self, smoothing=30):
        self.prev_gray  = None
        self.trajectory = []
        self.smoothing  = smoothing

    def _get_transform(self, prev, curr):
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

    def stabilize(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.trajectory.append(np.zeros(3))
            return frame
        m  = self._get_transform(self.prev_gray, gray)
        self.trajectory.append(np.array([m[0,2], m[1,2], np.arctan2(m[1,0], m[0,0])]))
        self.prev_gray = gray
        n      = len(self.trajectory)
        smooth = np.array(self.trajectory[max(0, n - self.smoothing//2):n]).mean(axis=0)
        diff   = smooth - self.trajectory[-1]
        h, w   = frame.shape[:2]
        fix    = np.array([[np.cos(diff[2]), -np.sin(diff[2]), diff[0]],
                           [np.sin(diff[2]),  np.cos(diff[2]), diff[1]]], dtype=np.float32)
        out = cv2.warpAffine(frame, fix, (w, h), borderMode=cv2.BORDER_REFLECT)
        cx, cy = int(w * 0.05), int(h * 0.05)
        return cv2.resize(out[cy:h-cy, cx:w-cx], (w, h))


# ════════════════════════════════════════════════════════════════════════════════
# YOLO
# ════════════════════════════════════════════════════════════════════════════════
def run_yolo(frame):
    img    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img    = cv2.resize(img, (input_w, input_h))
    tensor = (img / 255.0).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    return yolo_sess.run(None, {input_name: tensor})


def parse_yolo(outputs, img_h, img_w):
    box_output  = outputs[0]
    mask_output = outputs[1]
    num_classes = box_output.shape[1] - num_masks - 4
    predictions = box_output[0].T

    scores = np.max(predictions[:, 4:4+num_classes], axis=1)
    keep   = scores > CONF
    predictions, scores = predictions[keep], scores[keep]
    if len(predictions) == 0:
        return []

    class_ids  = np.argmax(predictions[:, 4:4+num_classes], axis=1)
    mask_preds = predictions[:, 4+num_classes:]

    cx, cy, w, h = predictions[:,0], predictions[:,1], predictions[:,2], predictions[:,3]
    sx, sy = img_w / input_w, img_h / input_h
    boxes  = np.stack([
        np.clip((cx - w/2) * sx, 0, img_w),
        np.clip((cy - h/2) * sy, 0, img_h),
        np.clip((cx + w/2) * sx, 0, img_w),
        np.clip((cy + h/2) * sy, 0, img_h),
    ], axis=1)

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF, IOU)
    if len(indices) == 0:
        return []
    indices    = indices.flatten()
    boxes      = boxes[indices]
    scores     = scores[indices]
    class_ids  = class_ids[indices]
    mask_preds = mask_preds[indices]

    proto         = mask_output[0]
    num_m, mh, mw = proto.shape
    masks = (1 / (1 + np.exp(-(mask_preds @ proto.reshape(num_m, -1))))).reshape(-1, mh, mw)

    boxes_mask = boxes.copy()
    boxes_mask[:, [0,2]] *= mw / img_w
    boxes_mask[:, [1,3]] *= mh / img_h
    blur_size = (int(img_w/mw), int(img_h/mh))

    detections = []
    for i in range(len(boxes)):
        x1,y1 = int(math.floor(boxes[i][0])), int(math.floor(boxes[i][1]))
        x2,y2 = int(math.ceil(boxes[i][2])),  int(math.ceil(boxes[i][3]))
        mx1,my1 = int(math.floor(boxes_mask[i][0])), int(math.floor(boxes_mask[i][1]))
        mx2,my2 = int(math.ceil(boxes_mask[i][2])),  int(math.ceil(boxes_mask[i][3]))

        crop = masks[i][my1:my2, mx1:mx2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (x2-x1, y2-y1), interpolation=cv2.INTER_CUBIC)
        crop = cv2.blur(crop, blur_size)
        crop = (crop > 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull    = cv2.convexHull(max(contours, key=cv2.contourArea))
        epsilon = 0.02 * cv2.arcLength(hull, True)
        while True:
            poly = cv2.approxPolyDP(hull, epsilon, True)
            if len(poly) <= 4:
                break
            epsilon *= 1.1
        poly = poly.reshape(-1, 2) + np.array([x1, y1])

        detections.append({
            "class":   LABELS[int(class_ids[i])],
            "score":   round(float(scores[i]), 3),
            "box":     [x1, y1, x2, y2],
            "polygon": poly.tolist(),
            "mask":    crop,
        })

    return detections


# ════════════════════════════════════════════════════════════════════════════════
# OCR
# ════════════════════════════════════════════════════════════════════════════════
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


def decode_ocr(preds):
    idx  = preds.argmax(axis=2)[0]
    text = ""
    for i in range(len(idx)):
        c = idx[i]
        if c != 0 and (i == 0 or c != idx[i-1]) and c < len(chars):
            text += chars[c]
    return text


def run_ocr(plate_img):
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

    return ""


# ════════════════════════════════════════════════════════════════════════════════
# DESSIN
# ════════════════════════════════════════════════════════════════════════════════
def draw(frame, detections, frame_count):
    out = frame.copy()
    for det in detections:
        color = COLORS[LABELS.index(det["class"])]
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class']} {det['score']}  {det.get('text', '')}"

        overlay = out.copy()
        overlay[y1:y2, x1:x2][det["mask"] == 1] = color
        out = cv2.addWeighted(out, 0.6, overlay, 0.4, 0)

        cv2.polylines(out, [np.array(det["polygon"], dtype=np.int32)],
                      isClosed=True, color=color, thickness=2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1-th-10), (x1+tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(out, f"frame {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return out


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════════════════════
def read_inputs(cap):
    ret, frame = cap.read()
    return ret, frame


def run_pipeline(frame, stabilizer):
    frame      = stabilizer.stabilize(frame)

    outputs    = run_yolo(frame)
    detections = parse_yolo(outputs, frame.shape[0], frame.shape[1])
    
    for det in detections:
        plate     = warp_plate(frame, det["polygon"]) if len(det["polygon"]) == 4 \
                    else frame[det["box"][1]:det["box"][3], det["box"][0]:det["box"][2]]
        
        ocr_raw_output =run_ocr(plate)
        det["text"] = clean_plate(ocr_raw_output)
    return frame, detections


def write_output(writer, frame, detections, frame_count):
    writer.write(draw(frame, detections, frame_count))

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
cap          = cv2.VideoCapture(INPUT_VIDEO)
fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Vidéo : {width}x{height} @ {fps}fps, {total_frames} frames")

writer      = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
stabilizer  = VideoStabilizer(smoothing=SMOOTHING)
frame_count = 0

while True:
    ret, frame = read_inputs(cap)
    if not ret:
        break

    frame, detections = run_pipeline(frame, stabilizer)
    write_output(writer, frame, detections, frame_count)

    frame_count += 1
    if frame_count % 100 == 0:
        texts = [d['text'] for d in detections if d.get('text')]
        pct   = f"{frame_count/total_frames*100:.1f}%" if total_frames > 0 else "?"
        print(f"  {frame_count} ({pct})  {texts}")

cap.release()
writer.release()
print(f"Terminé! {frame_count} frames → {OUTPUT_VIDEO}")