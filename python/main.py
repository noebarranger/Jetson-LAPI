import cv2
import os
from src import (
    get_image_stream, 
    get_video_stream, 
    save_image, 
    create_video_writer,
    init_inference_engines, 
    create_lapi_pipeline,
    draw_hud,
    create_stabilizer,
)

def main():
    
    
    YOLO_CAR = "../models/yolov8m.onnx"
    YOLO_PLATE = "../models/best.onnx"
    
    OCR_MODEL = "../models/ocr_model.onnx"
    OCR_DICT = "../models/en_dict.txt"

    
    INPUT_PATH = "../exemples/inputs/moto.mp4"
    OUTPUT_PATH = "../exemples/outputs/moto.mp4"
    
    is_video = INPUT_PATH.lower().endswith(('.mp4', '.avi', '.mov'))

    print("[INFO] Initialisation des moteurs TensorRT/CUDA...")
    run_car, run_plate, run_ocr = init_inference_engines(YOLO_CAR, YOLO_PLATE, OCR_MODEL, OCR_DICT)
    
    pipeline = create_lapi_pipeline(run_car, run_plate, run_ocr)
    stream = get_video_stream(INPUT_PATH) if is_video else get_image_stream(INPUT_PATH)
    stabilize = create_stabilizer(window_size=20)
    writer = None
    close_writer = None

    print(f"[INFO] Traitement lancé sur : {INPUT_PATH}")
    
    try:
        for packet in stream:
            frame = packet["image"]
            meta = packet["metadata"]

            processed_frame, text, kpts, car_box = pipeline(frame)
            
            res_img = draw_hud(
                    processed_frame, 
                    car_box, 
                    kpts, 
                    text,
                )
            if text:
                print(f"Frame {meta.get('frame_index', 1)}: Plaque détectée [{text}]", end="\r")

            if is_video:
                if writer is None:
                    writer, close_writer = create_video_writer(
                        OUTPUT_PATH, meta["fps"], meta["width"], meta["height"]
                    )
                writer(res_img)
            else:
                save_image(OUTPUT_PATH, res_img)
                
            
            
            

    except KeyboardInterrupt:
        print("\n[WARN] Interruption par l'utilisateur.")
    
    finally:
        if close_writer:
            close_writer()
        cv2.destroyAllWindows()
        print(f"\n[OK] Traitement terminé. Sortie : {OUTPUT_PATH}")

if __name__ == "__main__":
    main()