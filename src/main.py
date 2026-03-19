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


_HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_INPUT = os.path.join(_HERE, "../exemples/inputs/rush_2.avi")
VIDEO_OUTPUT = os.path.join(_HERE, OUTPUT_VIDEO)


meta   = get_video_meta(VIDEO_INPUT)
writer = open_video_writer(VIDEO_OUTPUT, meta["fps"], meta["width"], meta["height"])
source = frames_from_video(VIDEO_INPUT)

yolo = load_yolo(YOLO_MODEL, PROVIDERS)
ocr  = load_ocr(OCR_MODEL, CHARS_PATH, PROVIDERS)
state = make_initial_state()

for frame in source:
    frame, detections, state = run_pipeline(frame, state, yolo)
    print(state)
    annotated_frame = draw(frame, detections, 0)
    write_frame(writer, annotated_frame)
