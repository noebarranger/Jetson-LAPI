import cv2
import sys

def get_image_stream(path):
    """Charge une image unique et simule un flux d'un seul élément."""
    img = cv2.imread(path)
    if img is None:
        print(f"Erreur: Impossible de lire l'image {path}")
        return
    
    
    yield {
        "image": img,
        "metadata": {
            "width": img.shape[1],
            "height": img.shape[0],
            "fps": 1,
            "frame_index": 1,
            "total_frames": 1
        }
    }

def get_video_stream(path):
    """Générateur qui lit la vidéo frame par frame."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo {path}")
        return

    metadata = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            yield {
                "image": frame,
                "metadata": {**metadata, "frame_index": frame_idx}
            }
    finally:
        cap.release()

def create_video_writer(output_path, fps, width, height):
    """Retourne une fonction de fermeture (closure) pour écrire les frames."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    def write_frame(frame):
        if frame is not None:
            writer.write(frame)
            
    def close():
        writer.release()
        
    return write_frame, close

def save_image(output_path, frame):
    """Sauvegarde une image unique sur le disque."""
    if frame is not None:
        success = cv2.imwrite(output_path, frame)
        if success:
            print(f"Image enregistrée : {output_path}")
        else:
            print(f"Erreur lors de l'enregistrement de l'image : {output_path}")
    else:
        print(f"Erreur Pas de frame trouvé")