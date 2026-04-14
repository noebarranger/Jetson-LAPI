#!/usr/bin/env python3
from src.io.reader import frames_from_video, get_video_meta
from src.io.writer import write_frame, open_video_writer, close_video_writer
from src.io.viz    import draw
from src.config    import *
from src.models.ocr  import load_ocr
from src.models.yolo import load_yolo
from src.pipeline.core import make_initial_state, run_pipeline
from time import time
import os

_HERE        = os.path.dirname(os.path.abspath(__file__))
VIDEO_INPUT  = os.path.join(_HERE, "../exemples/inputs/rush_2.avi")
VIDEO_OUTPUT = os.path.join(_HERE, OUTPUT_VIDEO)

# ── Modèles ───────────────────────────────────────────────────────────────────
yolo  = load_yolo(YOLO_MODEL, PROVIDERS)
ocr   = load_ocr(OCR_MODEL, CHARS_PATH, PROVIDERS)
state = make_initial_state()

# ── Source + writer ───────────────────────────────────────────────────────────
meta   = get_video_meta(VIDEO_INPUT)
writer = open_video_writer(VIDEO_OUTPUT, meta["fps"], meta["width"], meta["height"])
source = frames_from_video(VIDEO_INPUT)

# ── Boucle vidéo ──────────────────────────────────────────────────────────────
frame_count  = 0
total_frames = meta["total"]

for frame in source:
    t0 = time()

    frame, detections, state = run_pipeline(frame, state, yolo, ocr)

    elapsed = time() - t0
    fps     = 1.0 / elapsed if elapsed > 0 else 0
    frame_count += 1

    annotated = draw(frame, detections, frame_count)
    write_frame(writer, annotated)
    texts = [d.get("text", "") for d in detections if d.get("text")]
    print(texts)


close_video_writer(writer)
print(f"Terminé! {frame_count} frames → {VIDEO_OUTPUT}")