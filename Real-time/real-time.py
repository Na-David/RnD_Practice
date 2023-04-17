import torch
import cv2
import os
import numpy as np
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import tkinter as tk
from tkinter import filedialog

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load DeepSORT configuration
cfg = get_config()
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(current_dir, "deep_sort.yaml")
cfg.merge_from_file(config_file)

checkpoint_path = os.path.join(
    current_dir, "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")

# Initialize DeepSORT
deepsort = DeepSort(
    checkpoint_path,
    max_dist=cfg.DEEPSORT.MAX_DIST,
    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg.DEEPSORT.MAX_AGE,
    n_init=cfg.DEEPSORT.N_INIT,
    nn_budget=cfg.DEEPSORT.NN_BUDGET,
    use_cuda=True
)

# Prompt user to select a video file
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select a video file")

cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Perform object tracking
    detections = results.xyxy[0].numpy()
    bbox_xywh = []
    confidences = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        bbox_xywh.append([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
        confidences.append(conf)

    bbox_xywh = np.array(bbox_xywh)
    class_ids = [results.names[int(cls)] for cls in detections[:, -1]]
    trackers = deepsort.update(bbox_xywh, confidences, frame)

    # Visualization
    for idx, track in enumerate(trackers):
        bbox_tlwh = deepsort._tlwh_to_xyxy(track)
        x1, y1, x2, y2 = bbox_tlwh
        class_id = class_ids[idx]
        track_id = track.track_id
        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"{class_id}-{track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Result", frame)

    # Wait for key input and exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
