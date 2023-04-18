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


def get_color(class_id):
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 128, 128), (0, 128, 0)
    ]
    return colors[class_id % len(colors)]


# Prompt user to select a video file
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select a video file")

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = 'output3.avi'
out = cv2.VideoWriter(output_video_path, fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

prev_trackers = {}
prev_trackers = {}  # Add this line to store previous frame's trackers
path_image = None  # Add this line to store the path image
# Initialize dictionary to store trajectories
trajectories = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Perform object tracking
    detections = results.pred[0].cpu().numpy()
    bbox_xywh = []
    confidences = []
    class_idxs = []

    # Initialize the path_image if not done yet
    if path_image is None:
        path_image = np.zeros_like(frame)

    for x1, y1, x2, y2, conf, cls in detections:
        bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        confidences.append(float(conf))
        class_idxs.append(int(cls))
    # Check if there are detections
    if len(bbox_xywh) > 0:
        # Convert bbox_xywh and confidences to numpy arrays
        bbox_xywh = np.array(bbox_xywh)
        confidences = np.array(confidences)

        trackers = deepsort.update(bbox_xywh, confidences, frame)
        print("trackers:", trackers)  # Add this line
    else:
        trackers = []
    # Visualization
    for idx, track in enumerate(trackers):
        x1, y1, x2, y2, track_id = track
        # class_id = results.names[class_idxs[idx]]

        # Check if the index is within the range of class_idxs
        if idx < len(classes):
            class_id = results.names[class_idxs[idx]]
        else:
            continue

        color = get_color(class_idxs[idx])

        if track_id in prev_trackers:
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_trackers[track_id]
            cv2.line(frame, (int((prev_x1 + prev_x2) / 2), int((prev_y1 + prev_y2) / 2)),
                     (int((x1 + x2) / 2), int((y1 + y2) / 2)), color, 2)

        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{class_id}-{track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update trajectory for the current track_id
        if track_id not in trajectories:
            trajectories[track_id] = [(int((x1+x2)/2), int((y1+y2)/2))]
        else:
            trajectories[track_id].append((int((x1+x2)/2), int((y1+y2)/2)))

        # Draw trajectory
        if len(trajectories[track_id]) > 1:
            for i in range(1, len(trajectories[track_id])):
                cv2.line(frame, trajectories[track_id][i-1],
                         trajectories[track_id][i], (0, 255, 0), 2)

        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"{class_id}-{track_id}", (int(x1),
                    int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Store current frame's trackers for the next frame
    prev_trackers = {track[-1]: track[:-1] for track in trackers}
    # Combine the path image and the current frame
    combined_frame = cv2.addWeighted(frame, 0.8, path_image, 0.2, 0)

    cv2.imshow("Result", frame)
    out.write(frame)

    # Wait for key input and exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
out.release()
cap.release()
cv2.destroyAllWindows()
