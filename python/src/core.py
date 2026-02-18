import cv2
import numpy as np
import onnxruntime as ort

def init_inference_engines(car_path, plate_path, ocr_path, dict_path):
    providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    
    car_sess = ort.InferenceSession(car_path, providers=providers)
    plate_sess = ort.InferenceSession(plate_path, providers=providers)
    ocr_sess = ort.InferenceSession(ocr_path, providers=providers)
    
    with open(dict_path, 'r', encoding='utf-8') as f:
        chars = ['blank'] + [line.strip() for line in f.readlines()] + [' ']
    
    def run_yolo_car(frame, conf_thresh=0.5):
        h_orig, w_orig = frame.shape[:2]
        blob = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 640))
        blob = blob.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32) / 255.0
        
        out = car_sess.run(None, {car_sess.get_inputs()[0].name: blob})[0]
        preds = np.squeeze(out).T 
        
        vehicle_indices = [6, 7, 9, 11] # all vehicules ids
        vehicle_scores = preds[:, vehicle_indices]
        
        max_vehicle_scores = np.max(vehicle_scores, axis=1)
        best_idx = np.argmax(max_vehicle_scores)
        
        if max_vehicle_scores[best_idx] > conf_thresh:
            box = preds[best_idx, :4] 
            
            x1 = (box[0] - box[2]/2) * w_orig / 640
            y1 = (box[1] - box[3]/2) * h_orig / 640
            x2 = (box[0] + box[2]/2) * w_orig / 640
            y2 = (box[1] + box[3]/2) * h_orig / 640
            return [int(x1), int(y1), int(x2), int(y2)]
        return None

    
    def run_yolo(frame, conf_thresh=0.4):
        h_orig, w_orig = frame.shape[:2]
        
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (640, 640))
        blob = resized.transpose((2, 0, 1))
        blob = blob[np.newaxis, ...].astype(np.float32) / 255.0
        
        
        out = plate_sess.run(None, {plate_sess.get_inputs()[0].name: blob})[0]
        preds = np.squeeze(out).T  

        
        
        kpts_confidences = (preds[:, 9] + preds[:, 12] + preds[:, 15] + preds[:, 18]) / 4
        
        best_idx = np.argmax(kpts_confidences)
        max_conf = kpts_confidences[best_idx]

        if max_conf > conf_thresh:
            print(f"[LAPI YOLO-M] Plaque trouvée ! Confiance Points: {max_conf:.2f}")
            
            
            kpts_raw = preds[best_idx, -12:]
            
            
            points = []
            for i in range(4):
                x = kpts_raw[i*3] * w_orig / 640
                y = kpts_raw[i*3+1] * h_orig / 640
                points.append([x, y])
            return points
            
        return None

    
    def run_ocr(plate_img):
        img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        new_w = int(48 * w / h)
        img = cv2.resize(img, (new_w, 48))
        img = (img.astype('float32') / 255.0 - 0.5) / 0.5
        img = img.transpose((2, 0, 1))[np.newaxis, :]
        
        preds = ocr_sess.run(None, {ocr_sess.get_inputs()[0].name: img})[0]
        idx = preds.argmax(axis=2)[0]
        
        res = ""
        for i in range(len(idx)):
            if idx[i] != 0 and (i == 0 or idx[i] != idx[i-1]):
                res += chars[idx[i]]
        return res
    
    return run_yolo_car, run_yolo,  run_ocr 


    