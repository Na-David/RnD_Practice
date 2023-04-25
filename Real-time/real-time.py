import torch
import cv2
import os
import numpy as np
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import tkinter as tk
from tkinter import filedialog

# Load YOLOv5 model with yolov5x version
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Load DeepSORT and merge
cfg = get_config()
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(current_dir, "deep_sort.yaml")
cfg.merge_from_file(config_file)

checkpoint_path = os.path.join(
    current_dir, "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")

# Configure DeepSORT
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

prev_trackers = {}
object_paths = {}
prev_frame_trackers = {}
path_image = None
trajectories = {}
color_map = {}

# Function to generate colors for objects


def get_color(idx):
    if idx not in color_map:
        np.random.default_rng()
        color_map[idx] = tuple(map(int, np.random.randint(0, 255, size=3)))
    return color_map[idx]


# Create object
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select a video file")

# Create Video Capture object and read selected video
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Create VideoWriter object to save the output
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    # get the object detection results
    detections = results.pred[0].cpu().numpy()
    bbox_xywh = []
    confidences = []
    class_idxs = []

    # initialize path_image (if it has not been initialized yet)
    if path_image is None:
        path_image = np.zeros_like(frame)

    # add information for each detected object to the corresponding list
    for x1, y1, x2, y2, conf, cls in detections:
        bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        confidences.append(float(conf))
        class_idxs.append(int(cls))

    # if there are any detected objects:
    if len(bbox_xywh) > 0:
        bbox_xywh = np.array(bbox_xywh)
        confidences = np.array(confidences)
        # update the trackers using the DeepSORT algorithm
        trackers = deepsort.update(bbox_xywh, confidences, frame)
    else:
        trackers = []

    # save the paths of all detected objects in the trackers
    for t in trackers:
        obj_id = int(t[4])
        if obj_id not in object_paths.keys():
            # initialize the object's path dictionary
            object_paths[obj_id] = []

    # Visualization
    # Iterate over all detected objects
    for idx, track in enumerate(trackers[:len(class_idxs)]):
        x1, y1, x2, y2, track_id = track
        class_idx = class_idxs[idx]
        class_id = results.names[class_idx]
        color = get_color(class_idx)

        # If the object was detected in the previous frame as well:
        if track_id in prev_trackers:
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_trackers[track_id]
            cv2.line(frame, (int((prev_x1 + prev_x2) / 2), int((prev_y1 + prev_y2) / 2)),
                     (int((x1 + x2) / 2), int((y1 + y2) / 2)), color, 2)

        # Draw bounding box and object ID on the current frame
        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{class_id}-{track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update object trajectory
        if track_id not in trajectories:
            trajectories[track_id] = [(int((x1+x2)/2), int((y1+y2)/2))]
        else:
            trajectories[track_id].append((int((x1+x2)/2), int((y1+y2)/2)))

        # Draw the object trajectory
        # If the length of the trajectory is greater than 1:
        if len(trajectories[track_id]) > 1:
            for i in range(1, len(trajectories[track_id])):
                cv2.line(frame, trajectories[track_id][i-1],
                         trajectories[track_id][i], (0, 255, 0), 2)  # prev_coordi, cur_coordi, color, thickness

    # Save current trackers as previous trackers
    prev_trackers = {track[-1]: track[:-1] for track in trackers}
    # Combine path image and current frame
    combined_frame = cv2.addWeighted(frame, 0.8, path_image, 0.2, 0)

    cv2.imshow("Result", frame)
    out.write(frame)

    # press 'q' to quite
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
out.release()
cap.release()
cv2.destroyAllWindows()
