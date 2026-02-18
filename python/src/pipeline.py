from .geometry import get_warped_plate
from .filters import create_stabilizer
import cv2

def create_lapi_pipeline(run_car, run_plate, run_ocr):
    
    stabilizer = create_stabilizer(window_size=10)

    def process_frame(frame):
        
        car_box = run_car(frame)
        
        kpts_absolute = None
        raw_text = None

        if car_box is not None:
            x1, y1, x2, y2 = car_box
            h_f, w_f = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_f, x2), min(h_f, y2)
            
            car_crop = frame[y1:y2, x1:x2]
            if car_crop.size > 0:
                
                kpts_relative = run_plate(car_crop)
                if kpts_relative is not None:
                    kpts_absolute = [[p[0] + x1, p[1] + y1] for p in kpts_relative]
                    
                    
                    plate_img = get_warped_plate(car_crop, kpts_relative)
                    raw_text = run_ocr(plate_img)

        
        
        stable_text, stable_kpts, stable_car_box = stabilizer(raw_text, kpts_absolute, car_box)

        return frame, stable_text, stable_kpts, stable_car_box

    return process_frame