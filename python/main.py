import cv2
import os
import time

from src import (
    get_image_stream, 
    get_video_stream, 
    save_image, 
    create_video_writer,
    init_inference_engines, 
    create_lapi_pipeline,
    draw_tracks
)

def main():
    YOLO_CAR = "../models/yolov8s.onnx"
    YOLO_PLATE = "../models/yolov8s-pose.onnx"
    
    OCR_MODEL = "../models/ocr_model.onnx"
    OCR_DICT = "../models/en_dict.txt"

    
    INPUT_PATH = "../exemples/inputs/6.mp4"
    OUTPUT_PATH = "../exemples/outputs/6.mp4"
    
    is_video = INPUT_PATH.lower().endswith(('.mp4', '.avi', '.mov'))

    print("[INFO] Initialisation des moteurs TensorRT/CUDA...")
    run_car, run_plate, run_ocr = init_inference_engines(YOLO_CAR, YOLO_PLATE, OCR_MODEL, OCR_DICT)
    
    pipeline = create_lapi_pipeline(run_car, run_plate, run_ocr)
    stream = get_video_stream(INPUT_PATH) if is_video else get_image_stream(INPUT_PATH)
    writer = None
    close_writer = None

    print(f"[INFO] Traitement lancé sur : {INPUT_PATH}")
    start = time.time()
    try:
        for packet in stream:
            frame = packet["image"]
            meta = packet["metadata"]

            tracks = pipeline(frame)
            
            # ← DEBUG
            print(f"[Main] Type tracks: {type(tracks)}, len: {len(tracks)}")
            if tracks:
                print(f"[Main] Premier track: ID={tracks[0].id}, "
                    f"vehicle_box={tracks[0].vehicle_box}, "
                    f"has_plate={tracks[0].has_plate}")
            
            res_img = draw_tracks(frame, tracks)
            
            # ← DEBUG
            print(f"[Main] Type res_img: {type(res_img)}, "
                f"shape: {res_img.shape if hasattr(res_img, 'shape') else 'N/A'}")    
            
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
    stop = time.time() - start            
    print(stop) 

if __name__ == "__main__":
    main()