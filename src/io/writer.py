from .viz import draw

def write_output(writer, frame, detections, frame_count):
    writer.write(draw(frame, detections, frame_count))

import os
import cv2


# ── Vidéo fichier ─────────────────────────────────────────────────────────────
def open_video_writer(path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    handle = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not handle.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir le writer : {path}")
    return handle


def write_frame(handle, frame):
    handle.write(frame)


def close_video_writer(handle):
    handle.release()


# ── Affichage écran ────────────────────────────────────────────────────────────
def display(window, frame):
    """
    Affiche la frame dans une fenêtre.
    Retourne False si l'utilisateur appuie sur 'q'.
    """
    cv2.imshow(window, frame)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def close_display():
    cv2.destroyAllWindows()


def make_winrate_state():
    return {"total": 0, "correct": 0, "results": []}


def normalize(text):
    """Retire espaces et met en majuscule pour comparaison."""
    return text.upper().replace(" ", "").strip()


def update_winrate(state, ground_truth, detected):
    """
    Compare ground_truth et detected.
    Retourne (new_state, is_correct).
    """
    gt  = normalize(ground_truth)
    det = normalize(detected)

    is_correct = gt == det
    new_total  = state["total"] + 1
    new_correct = state["correct"] + (1 if is_correct else 0)
    winrate    = new_correct / new_total * 100

    result = {
        "image":    state["total"],
        "expected": gt,
        "detected": det,
        "correct":  is_correct,
    }

    # affichage terminal
    icon = "OK" if is_correct else "ERROR"
    print(
        f"  {icon} [{new_total:5d}]  "
        f"expected={gt:<10}  "
        f"detected={det:<10}  "
        f"winrate={winrate:.1f}%",
        flush=True
    )

    return {
        **state,
        "total":   new_total,
        "correct": new_correct,
        "results": state["results"] + [result],
    }


def print_summary(state):
    """Affiche le résumé final."""
    total   = state["total"]
    correct = state["correct"]
    winrate = correct / total * 100 if total > 0 else 0

    print("\n" + "═" * 55)
    print(f"  RÉSULTAT FINAL")
    print(f"  Images testées : {total}")
    print(f"  Correctes      : {correct}")
    print(f"  Winrate        : {winrate:.1f}%")
    print("═" * 55)