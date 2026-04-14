import math
import re
import cv2
import numpy as np
import onnxruntime as ort
from ..config import YOLO_MODEL, LABELS, CONF, IOU


def load_yolo(path, providers):
    sess       = ort.InferenceSession(path, providers=providers)
    input_name = sess.get_inputs()[0].name
    input_h    = sess.get_inputs()[0].shape[2]
    input_w    = sess.get_inputs()[0].shape[3]
    print(f"YOLO : {sess.get_providers()}")
    return {
        "sess":       sess,
        "input_name": input_name,
        "input_h":    input_h,
        "input_w":    input_w,
        "num_masks":  32,
    }

def run_yolo(yolo_sess, frame, input_w, input_h, input_name):
    img    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img    = cv2.resize(img, (input_w, input_h))
    tensor = (img / 255.0).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    return yolo_sess.run(None, {input_name: tensor})

def _process_mask(crop, blur_size):
    crop = cv2.blur(crop, blur_size)
    crop = (crop > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # hull sur TOUS les contours → pas seulement le plus grand
    all_points = np.vstack(contours)
    hull       = cv2.convexHull(all_points)

    epsilon = 0.02 * cv2.arcLength(hull, True)
    while True:
        poly = cv2.approxPolyDP(hull, epsilon, True)
        if len(poly) <= 4:
            break
        epsilon *= 1.1

    return crop, poly

def _extract_detections(boxes, scores, class_ids, mask_preds, proto, img_h, img_w):
    """Calcule les masques et extrait les polygones."""
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

        crop, poly = _process_mask(crop, blur_size)
        if poly is None:
            continue

        poly = poly.reshape(-1, 2) + np.array([x1, y1])

        detections.append({
            "class":   LABELS[int(class_ids[i])],
            "score":   round(float(scores[i]), 3),
            "box":     [x1, y1, x2, y2],
            "polygon": poly.tolist(),
            "mask":    crop,
        })

    return detections


def parse_yolo(outputs, img_h, img_w, input_h, input_w, num_masks):
    """Parse les sorties brutes YOLO → liste de détections."""
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
    indices = indices.flatten()

    return _extract_detections(
        boxes[indices], scores[indices],
        class_ids[indices], mask_preds[indices],
        mask_output[0], img_h, img_w
    )