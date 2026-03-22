

import cv2
import json
import os
def frames_from_dataset(images_dir, annotations_path):
    """
    Générateur qui yield (frame, ground_truth) pour chaque image annotée.
 
    ground_truth = {
        "image_name": "frame_07424.jpg",
        "polygon":    [[x,y], ...],
        "plate_id":   "K81VRC",
    }
    """
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
 
    for ann in annotations:
        filename = os.path.basename(ann["image_name"])
        path     = os.path.join(images_dir, filename)
 
        frame = cv2.imread(path)
        if frame is None:
            print(f"image manquante : {path}")
            continue
 
        yield frame, {
            "image_name": filename,
            "polygon":    ann["polygon"],
            "plate_id":   ann["plate_id"],
        }

def frames_from_image(path):
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Image introuvable : {path}")
    yield frame


def frames_from_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Vidéo introuvable : {path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def frames_from_gstreamer(width=1280, height=720, fps=30, sensor_id=0, flip=0):
    pipeline = (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("GStreamer indisponible")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def get_video_meta(path):
    """Retourne les métadonnées sans ouvrir le flux de frames."""
    cap = cv2.VideoCapture(path)
    meta = {
        "fps":    int(cap.get(cv2.CAP_PROP_FPS)) or 30,
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total":  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta

