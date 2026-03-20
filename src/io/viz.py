import cv2
from src.config import COLORS, LABELS
import numpy as np

def draw(frame, detections, frame_count):
    out = frame.copy()
    for det in detections:
        color = COLORS[0]
        x1, y1, x2, y2 = det["box"]
        label = f" {det['score']}  {det.get('text', '')}"

        overlay = out.copy()
        overlay[y1:y2, x1:x2][det["mask"] == 1] = color
        out = cv2.addWeighted(out, 0.6, overlay, 0.4, 0)

        cv2.polylines(out, [np.array(det["polygon"], dtype=np.int32)],
                      isClosed=True, color=color, thickness=2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1-th-10), (x1+tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(out, f"frame {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return out