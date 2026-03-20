import os

# ── Environnement ─────────────────────────────────────────────────────────────
# docker : docker run -e ENV=docker ...
# local  : python directement sur Jetson ou env conda
_ENV = os.getenv("ENV", "local")

# ── Chemins ───────────────────────────────────────────────────────────────────
if _ENV == "docker":
    YOLO_MODEL   = "/app/src/models_weight/best.onnx"
    OCR_MODEL    = "/app/src/models_weight/ocr_model.onnx"
    CHARS_PATH   = "/app/src/models_weight/en_dict.txt"
    INPUT_VIDEO  = "/app/src/videos/rush_2.avi"
    OUTPUT_VIDEO = "../exemples/outputs/rush_2.avi"
else:
    YOLO_MODEL   = "../src/models_weight/best.onnx"
    OCR_MODEL    = "../src/models_weight/ocr_model.onnx"
    CHARS_PATH   = "../src/models_weight/en_dict.txt"
    INPUT_VIDEO  = "../src/videos/rush_2.avi"
    OUTPUT_VIDEO = "../exemples/outputs/rush_2.avi"

# ── Inférence ─────────────────────────────────────────────────────────────────
CONF      = 0.5
IOU       = 0.5
SMOOTHING = 100
LABELS    = ["day", "night"]
COLORS    = [(0, 255, 0), (0, 0, 255)]
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]