from .viz import draw

def write_output(writer, frame, detections, frame_count):
    writer.write(draw(frame, detections, frame_count))


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

