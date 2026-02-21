import cv2
import numpy as np
import onnxruntime as ort
from .datastruct import Detection,PlateDetection
from typing import Tuple

def init_inference_engines(car_path, plate_path, ocr_path, dict_path):
    providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    
    car_sess = ort.InferenceSession(car_path, providers=providers)
    plate_sess = ort.InferenceSession(plate_path, providers=providers)
    ocr_sess = ort.InferenceSession(ocr_path, providers=providers)
    
    with open(dict_path, 'r', encoding='utf-8') as f:
        chars = ['blank'] + [line.strip() for line in f.readlines()] + [' ']
    
    def run_yolo_car(frame, conf_thresh=0.5, iou_thresh=0.45):
        h_orig, w_orig = frame.shape[:2]
        blob = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 640))
        blob = blob.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32) / 255.0

        out = car_sess.run(None, {car_sess.get_inputs()[0].name: blob})[0]
        preds = np.squeeze(out).T 

        # Classes véhicules
        vehicle_indices = [6, 7, 9, 11] 
        vehicle_scores = preds[:, vehicle_indices]
        max_scores = np.max(vehicle_scores, axis=1)

        # Filtrage par confiance
        candidate_indices = np.where(max_scores > conf_thresh)[0]

        boxes = []
        scores = []
        for idx in candidate_indices:
            box = preds[idx, :4]
            w = box[2]
            h = box[3]
            x = box[0] - w/2
            y = box[1] - h/2
            boxes.append([float(x), float(y), float(w), float(h)])
            scores.append(float(max_scores[idx]))

        # NMS
        nms_indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)

        final_cars = []
        if len(nms_indices) > 0:
            for i in nms_indices.flatten():
                x, y, w, h = boxes[i]
                x1 = int(x * w_orig / 640)
                y1 = int(y * h_orig / 640)
                x2 = int((x + w) * w_orig / 640)
                y2 = int((y + h) * h_orig / 640)

                final_cars.append(Detection(
                    box=(x1, y1, x2, y2),
                    confidence=float(scores[i]),
                    class_id=0  # 0 = voiture
                ))
                
        return final_cars
    
    def run_yolo(frame, conf_thresh=0.4):
        h_orig, w_orig = frame.shape[:2]
        blob = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 640))
        blob = blob.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32) / 255.0

        out = plate_sess.run(None, {plate_sess.get_inputs()[0].name: blob})[0]
        preds = np.squeeze(out).T  

        kpts_confidences = (preds[:, 8] + preds[:, 11] + preds[:, 14] + preds[:, 17]) / 4
        best_idx = np.argmax(kpts_confidences)

        if kpts_confidences[best_idx] > conf_thresh:
            raw_indices = [6, 7, 9, 10, 12, 13, 15, 16]
            vals = preds[best_idx, raw_indices]
            
            # 4 points en tuples (immutable)
            points = []
            for i in range(0, 8, 2):
                x = vals[i] * w_orig / 640
                y = vals[i+1] * h_orig / 640
                points.append((float(x), float(y)))
            
            # Calculer dimensions approximatives depuis les 4 points
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plate_w = max(xs) - min(xs)
            plate_h = max(ys) - min(ys)
            
            MIN_PLATE_WIDTH = 48   # px
            MIN_PLATE_HEIGHT = 24   # px
            
            if plate_w >= MIN_PLATE_WIDTH and plate_h >= MIN_PLATE_HEIGHT:
                return PlateDetection(
                    keypoints_rel=(points[0], points[1], points[2], points[3]),
                    confidence=float(kpts_confidences[best_idx])
                )
            else:
                # Plaque trop petite, ignorée
                print(f"[PlateDet] Plaque rejetée: {plate_w:.0f}x{plate_h:.0f} "
                    f"(min {MIN_PLATE_WIDTH}x{MIN_PLATE_HEIGHT})")
        
        return None

    
    def run_ocr(plate_img):
        # 1. PRÉTRAITEMENT CLAHE (normalisation d'histogramme)
        plate_enhanced = apply_clahe(plate_img, clip_limit=3.0, tile_grid_size=(8, 8))
        
        # 2. CONVERSION BGR => RGB
        img = cv2.cvtColor(plate_enhanced, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # 3. RESIZE PP-OCRv4 : hauteur fixe 48px, largeur proportionnelle (max 320)
        rec_image_shape = [3, 48, 320]  # [C, H, W] standard PP-OCRv4
        imgC, imgH, imgW = rec_image_shape
        
        # Calcul ratio pour garder proportions
        ratio = w / float(h)
        resized_w = min(int(imgH * ratio), imgW)  # Limiter à 320px max (max de pp-ocrv4)
        
        # Resize à la hauteur 48px, largeur calculée
        resized_image = cv2.resize(img, (resized_w, imgH))
        
        # 4. NORMALISATION [-1, 1] (PP-OCRv4 standard)
        resized_image = resized_image.astype('float32') / 255.0
        resized_image = (resized_image - 0.5) / 0.5  # Équivalent à [-1, 1]
        
        # 5. PADDING pour atteindre exactement 320px de largeur
        # Créer tensor final [3, 48, 320] avec padding à droite
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        resized_image = resized_image.transpose((2, 0, 1))  # HWC → CHW
        padding_im[:, :, 0:resized_w] = resized_image
        
        # 6. INFÉRENCE
        preds = ocr_sess.run(None, {ocr_sess.get_inputs()[0].name: padding_im[np.newaxis, :]})[0]
        idx = preds.argmax(axis=2)[0]
        
        # 7. DÉCODAGE CTC 
        res = ""
        for i in range(len(idx)):
            if idx[i] != 0 and (i == 0 or idx[i] != idx[i-1]):
                res += chars[idx[i]]
        
        return res
    return run_yolo_car, run_yolo,  run_ocr

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, 
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applique CLAHE pour améliorer le contraste local.
    
    Args:
        image: Image BGR ou RGB
        clip_limit: Seuil de clipping (plus élevé = plus de contraste)
        tile_grid_size: Taille des tuiles pour l'égalisation adaptative
    
    Returns:
        Image avec contraste amélioré
    """
    # Convertir en LAB (L = luminance, A/B = couleur)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Séparer les canaux
    l, a, b = cv2.split(lab)
    
    # Appliquer CLAHE sur le canal L (luminance)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    
    # Recombiner les canaux
    lab_clahe = cv2.merge([l_clahe, a, b])
    
    # Retourner en BGR
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return result