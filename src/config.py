import os

# ── Environnement ─────────────────────────────────────────────────────────────
# docker : docker run -e ENV=docker ...
# local  : python directement sur Jetson ou env conda
_ENV = os.getenv("ENV", "local")

# ── Chemins ───────────────────────────────────────────────────────────────────
if _ENV == "docker":
    YOLO_MODEL   = "/app/runs/train_merged6/weights/best.onnx"
    OCR_MODEL    = "/app/ocr/ocr_model.onnx"
    CHARS_PATH   = "/app/ocr/en_dict.txt"
    INPUT_VIDEO  = "/app/rush_2.avi"
    OUTPUT_VIDEO = "../exemples/outputs/rush_2.avi"
else:
    YOLO_MODEL   = "../plate-detection/models_weights/best.onnx"
    OCR_MODEL    = "../plate-detection/models_weights/ocr_model.onnx"
    CHARS_PATH   = "../plate-detection/models_weights/en_dict.txt"
    INPUT_VIDEO  = "../plate-detection/videos/rush_2.avi"
    OUTPUT_VIDEO = "../exemples/outputs/rush_2.avi"

# ── Inférence ─────────────────────────────────────────────────────────────────
CONF      = 0.5
IOU       = 0.5
SMOOTHING = 100
LABELS    = ["day", "night"]
COLORS    = [(0, 255, 0), (0, 0, 255)]
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]