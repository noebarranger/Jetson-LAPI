#!/usr/bin/env python3
from src.io.reader import frames_from_video, frames_from_dataset
from src.io.writer import display, close_display, write_frame, open_video_writer, print_summary, make_metrics_state, update_metrics
import os
from src.io.reader import frames_from_video, get_video_meta
from src.io.viz import draw
from src.config import *
import cv2
from src.models.ocr import load_ocr
from src.models.yolo import load_yolo
from src.pipeline.core import make_initial_state, run_pipeline, make_pipeline_metrics, update_pipeline_metrics, print_pipeline_metrics
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
metrics = make_metrics_state()
benchmark = make_pipeline_metrics()

frame_count = 0
total_time = 0
for frame, ground_truth in source:
    t0 = time()
    frame, detections, state, times = run_pipeline(frame, state, yolo, ocr)
    benchmark = update_pipeline_metrics(benchmark, times)
    elapsed = time() - t0
    fps     = 1.0 / elapsed if elapsed > 0 else 0
    frame_count += 1
    total_time += elapsed

    gt = ground_truth["plate_id"].upper().replace(" ", "")

    # garder la détection la plus proche du ground truth
    best = ""
    if detections:
        texts = [d.get("text", "") for d in detections if d.get("text")]
        if texts:
            best = max(texts, key=lambda t: sum(a==b for a,b in zip(t, gt)))

    metrics = update_metrics(metrics, gt, best)
    
    # ── Affichage des métriques ──
    avg_fps = frame_count / total_time if total_time > 0 else 0
    status = "✓" if best == gt else "✗"
    print(f"[{status}] {frame_count:3d} | {elapsed:.3f}s | FPS: {fps:.1f} | Avg: {avg_fps:.1f} fps | GT={gt} | DET={best}")
    
    if best != gt:
        annotated = draw(frame, detections, frame_count)
        label     = f"GT={gt}  DET={best}"
        cv2.putText(annotated, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        filename = f"{frame_count:05d}_gt{gt}_det{best}.jpg"
        cv2.imwrite(os.path.join(ERRORS_DIR, filename), annotated)

print("\n" + "="*60)
print(f"✅ Traitement terminé: {frame_count} images en {total_time:.1f}s")
print(f"📊 FPS moyen: {frame_count/total_time if total_time > 0 else 0:.1f} fps")
print("="*60)
print_summary(metrics)

# ── Affichage du benchmark détaillé ────────────────────────────────────────
print_pipeline_metrics(benchmark)
