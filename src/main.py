#!/usr/bin/env python3
from src.io.reader import frames_from_video
from src.io.writer import display, close_display, write_frame, open_video_writer
import os
from src.io.reader import frames_from_video, get_video_meta
from src.io.viz import draw
from src.config import *

from src.models.ocr import load_ocr
from src.models.yolo import load_yolo
from src.pipeline.core import make_initial_state, run_pipeline
from time import time

_HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_INPUT = os.path.join(_HERE, "../exemples/inputs/rush_2.avi")
VIDEO_OUTPUT = os.path.join(_HERE, OUTPUT_VIDEO)


meta   = get_video_meta(VIDEO_INPUT)
writer = open_video_writer(VIDEO_OUTPUT, meta["fps"], meta["width"], meta["height"])
source = frames_from_video(VIDEO_INPUT)

yolo = load_yolo(YOLO_MODEL, PROVIDERS)
ocr  = load_ocr(OCR_MODEL, CHARS_PATH, PROVIDERS)
state = make_initial_state()


frame_count = 0

for frame in source:
    t0 = time()

    frame, detections, state = run_pipeline(frame, state, yolo, ocr)

    elapsed = time() - t0
    fps     = 1.0 / elapsed if elapsed > 0 else 0

    frame_count += 1
    print(f"frame {frame_count:5d} | {elapsed*1000:.1f} ms | {fps:.1f} fps | {detections} det")

    annotated_frame = draw(frame, detections, frame_count)
    write_frame(writer, annotated_frame)

