#!/usr/bin/env python3
from src.io.reader import frames_from_video, get_video_meta
from src.io.writer import write_frame, open_video_writer, close_video_writer
from src.io.viz    import draw
from src.config    import *
from src.models.ocr  import load_ocr
from src.models.yolo import load_yolo
from src.pipeline.core import make_initial_state, run_pipeline, make_pipeline_metrics, update_pipeline_metrics, print_pipeline_metrics
from time import time
import os

_HERE        = os.path.dirname(os.path.abspath(__file__))
VIDEO_INPUT  = os.path.join(_HERE, "../exemples/inputs/rush_2.avi")
VIDEO_OUTPUT = os.path.join(_HERE, OUTPUT_VIDEO)

# ── Modèles ───────────────────────────────────────────────────────────────────
yolo  = load_yolo(YOLO_MODEL, PROVIDERS)
ocr   = load_ocr(OCR_MODEL, CHARS_PATH, PROVIDERS)
state = make_initial_state()
benchmark = make_pipeline_metrics()

# ── Source + writer ───────────────────────────────────────────────────────────
meta   = get_video_meta(VIDEO_INPUT)
writer = open_video_writer(VIDEO_OUTPUT, meta["fps"], meta["width"], meta["height"])
source = frames_from_video(VIDEO_INPUT)

# ── Boucle vidéo ──────────────────────────────────────────────────────────────
frame_count  = 0
total_frames = meta["total"]
total_time   = 0

for frame in source:
    t0 = time()

    frame, detections, state, times = run_pipeline(frame, state, yolo, ocr)
    benchmark = update_pipeline_metrics(benchmark, times)

    elapsed = time() - t0
    fps     = 1.0 / elapsed if elapsed > 0 else 0
    frame_count += 1
    total_time += elapsed

    annotated = draw(frame, detections, frame_count)
    write_frame(writer, annotated)
    texts = [d.get("text", "") for d in detections if d.get("text")]
    
    # ── Affichage des métriques ──
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"[{frame_count:3d}/{total_frames}] {elapsed:.3f}s | FPS: {fps:.1f} | Avg: {avg_fps:.1f} fps | {texts}")


close_video_writer(writer)
avg_fps = frame_count / total_time if total_time > 0 else 0
print(f"\n✅ Terminé! {frame_count} frames en {total_time:.1f}s → {VIDEO_OUTPUT}")
print(f"📊 FPS moyen: {avg_fps:.1f} fps")

# ── Affichage du benchmark détaillé ────────────────────────────────────────
print_pipeline_metrics(benchmark)