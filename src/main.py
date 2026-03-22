#!/usr/bin/env python3
from src.io.reader import frames_from_video, frames_from_dataset
from src.io.writer import display, close_display, write_frame, open_video_writer, make_winrate_state, update_winrate, print_summary
import os
from src.io.reader import frames_from_video, get_video_meta
from src.io.viz import draw
from src.config import *
import cv2
from src.models.ocr import load_ocr
from src.models.yolo import load_yolo
from src.pipeline.core import make_initial_state, run_pipeline
from time import time

_HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_INPUT = os.path.join(_HERE, "../exemples/inputs/rush_2.avi")
VIDEO_OUTPUT = os.path.join(_HERE, OUTPUT_VIDEO)
ERRORS_DIR = os.path.join(_HERE, "../exemples/outputs/errors")
os.makedirs(ERRORS_DIR, exist_ok=True)


meta   = get_video_meta(VIDEO_INPUT)
writer = open_video_writer(VIDEO_OUTPUT, meta["fps"], meta["width"], meta["height"])
source = frames_from_video(VIDEO_INPUT)
IMAGES_DIR       = os.path.join(_HERE, "../exemples/inputs/testset/images")
ANNOTATIONS_PATH = os.path.join(_HERE, "../exemples/inputs/testset/annotations.json")

source       = frames_from_dataset(IMAGES_DIR, ANNOTATIONS_PATH)
yolo = load_yolo(YOLO_MODEL, PROVIDERS)
ocr  = load_ocr(OCR_MODEL, CHARS_PATH, PROVIDERS)

state = make_initial_state()
winrate_state = make_winrate_state()

frame_count = 0


for frame, ground_truth in source:
    t0 = time()
    frame, detections, state = run_pipeline(frame, state, yolo, ocr)
    elapsed = time() - t0
    fps     = 1.0 / elapsed if elapsed > 0 else 0
    frame_count += 1

    detected      = detections[0]["text"] if detections else ""
    winrate_state = update_winrate(winrate_state, ground_truth["plate_id"], detected)

    # sauvegarder les erreurs avec HUD
    if detected != ground_truth["plate_id"].upper().replace(" ", ""):
        annotated = draw(frame, detections, frame_count)

        # HUD erreur
        label = f"GT={ground_truth['plate_id']}  DET={detected}"
        cv2.putText(annotated, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        filename = f"{frame_count:05d}_gt{ground_truth['plate_id']}_det{detected}.jpg"
        cv2.imwrite(os.path.join(ERRORS_DIR, filename), annotated)

print_summary(winrate_state)

