import os
import cv2
from collections import defaultdict
from .viz import draw


# ════════════════════════════════════════════════════════════════════════════════
# VIDEO
# ════════════════════════════════════════════════════════════════════════════════

def open_video_writer(path, fps, width, height):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    handle = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not handle.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir le writer : {path}")
    return handle

def write_frame(handle, frame):
    handle.write(frame)

def close_video_writer(handle):
    handle.release()

def write_output(writer, frame, detections, frame_count):
    writer.write(draw(frame, detections, frame_count))


# ════════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ════════════════════════════════════════════════════════════════════════════════

def display(window, frame):
    cv2.imshow(window, frame)
    return (cv2.waitKey(1) & 0xFF) != ord("q")

def close_display():
    cv2.destroyAllWindows()


# ════════════════════════════════════════════════════════════════════════════════
# METRICS STATE
# ════════════════════════════════════════════════════════════════════════════════



def normalize(text):
    return text.upper().replace(" ", "").strip()


def make_metrics_state():
    return {
        # plaques
        "TP": 0, "TN": 0, "FP": 0, "FN": 0,
        # caractères
        "char_TP": 0,   # char correct à la bonne position
        "char_FP": 0,   # char détecté mais mauvais
        "char_FN": 0,   # char attendu mais manquant
        "results": [],
    }


def _char_confusion(gt, det):
    """Retourne (TP, FP, FN) au niveau caractère."""
    char_TP = sum(1 for i in range(min(len(gt), len(det))) if gt[i] == det[i])
    char_FP = sum(1 for i in range(min(len(gt), len(det))) if gt[i] != det[i])
    char_FN = max(0, len(gt) - len(det))   # caractères manquants
    return char_TP, char_FP, char_FN


def update_metrics(state, ground_truth, detected):
    gt  = normalize(ground_truth)
    det = normalize(detected)

    has_gt  = gt  != ""
    has_det = det != ""

    if has_gt and has_det and gt == det:
        case = "TP"
    elif not has_gt and not has_det:
        case = "TN"
    elif has_det and gt != det:
        case = "FP"
    else:
        case = "FN"

    char_TP, char_FP, char_FN = _char_confusion(gt, det)

    new_state = {
        **state,
        case:        state[case] + 1,
        "char_TP":   state["char_TP"] + char_TP,
        "char_FP":   state["char_FP"] + char_FP,
        "char_FN":   state["char_FN"] + char_FN,
        "results":   state["results"] + [{
            "expected": gt, "detected": det, "case": case,
            "char_TP": char_TP, "char_FP": char_FP, "char_FN": char_FN,
        }],
    }

    TP, FP, FN = new_state["TP"], new_state["FP"], new_state["FN"]
    total      = TP + FP + FN + new_state["TN"]
    precision  = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall     = TP / (TP + FN) if (TP + FN) > 0 else 0

    icon = "OK   " if case == "TP" else case
    print(
        f"  {icon} [{total:5d}]  "
        f"expected={gt:<10}  detected={det:<10}  "
        f"P={precision:.2f}  R={recall:.2f}",
        flush=True
    )
    return new_state


def print_summary(state):
    TP, TN = state["TP"], state["TN"]
    FP, FN = state["FP"], state["FN"]
    total  = TP + TN + FP + FN

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (TP + TN) / total * 100 if total > 0 else 0

    cTP, cFP, cFN = state["char_TP"], state["char_FP"], state["char_FN"]
    c_precision = cTP / (cTP + cFP) if (cTP + cFP) > 0 else 0
    c_recall    = cTP / (cTP + cFN) if (cTP + cFN) > 0 else 0
    c_f1        = 2 * c_precision * c_recall / (c_precision + c_recall) if (c_precision + c_recall) > 0 else 0
    c_accuracy  = cTP / (cTP + cFP + cFN) * 100 if (cTP + cFP + cFN) > 0 else 0

    print("\n" + "═" * 55)
    print(f"  RÉSULTAT FINAL")
    print(f"\n  — PLAQUES —")
    print(f"  Total      : {total}")
    print(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
    print(f"  Accuracy   : {accuracy:.1f}%")
    print(f"  Precision  : {precision:.3f}")
    print(f"  Recall     : {recall:.3f}")
    print(f"  F1 score   : {f1:.3f}")
    print(f"\n  — CARACTÈRES —")
    print(f"  TP={cTP}  FP={cFP}  FN={cFN}")
    print(f"  Accuracy   : {c_accuracy:.1f}%")
    print(f"  Precision  : {c_precision:.3f}")
    print(f"  Recall     : {c_recall:.3f}")
    print(f"  F1 score   : {c_f1:.3f}")
    print("═" * 55)