import os

# ── Environnement ─────────────────────────────────────────────────────────────
# docker : docker run -e ENV=docker ...
# local  : python directement sur Jetson ou env conda
_ENV = os.getenv("ENV", "local")

# ── Chemins absolus (depuis le répertoire src/) ────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if _ENV == "docker":
    YOLO_MODEL   = "/app/src/models_weight/best.onnx"
    OCR_MODEL    = "/app/src/models_weight/ocr_model.onnx"
    CHARS_PATH   = "/app/src/models_weight/en_dict.txt"
    INPUT_VIDEO  = "/app/exemples/inputs/rush_2.avi"
    OUTPUT_VIDEO = "/app/exemples/outputs/rush_2.avi"
else:
    # Chemins absolus pour "local" (Jetson ou PC)
    YOLO_MODEL   = os.path.join(_SRC_DIR, "models_weight", "best.onnx")
    OCR_MODEL    = os.path.join(_SRC_DIR, "models_weight", "ocr_model.onnx")
    CHARS_PATH   = os.path.join(_SRC_DIR, "models_weight", "en_dict.txt")
    INPUT_VIDEO  = os.path.join(_PROJECT_ROOT, "exemples", "inputs", "rush_2.avi")
    OUTPUT_VIDEO = os.path.join(_PROJECT_ROOT, "exemples", "outputs", "rush_2.avi")

# ── Inférence ─────────────────────────────────────────────────────────────────
CONF      = 0.5
IOU       = 0.5
SMOOTHING = 100
LABELS    = ["day", "night"]
COLORS    = [(0, 255, 0), (0, 0, 255)]
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]