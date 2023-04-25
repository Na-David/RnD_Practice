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

prev_trackers = {}  # Dictionary for trackers (path tracking)
object_paths = {}  # Dictionary to track object movement paths
prev_frame_trackers = {}  # Dictionary to track path in the previous frame
path_image = None  # Initialize for storing path image
trajectories = {}  # Dictionary to store object trajectories
color_map = {}  # Dictionary to store object colors

# Function to generate colors for objects


def get_color(idx):
    if idx not in color_map:
        # Initialize default random number generator
        np.random.default_rng()
        # Generate 3 random integers between 0 and 224 and return as tuple
        color_map[idx] = tuple(map(int, np.random.randint(0, 255, size=3)))
    return color_map[idx]


# Create object
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select a video file")

# Create Video Capture object and read selected video
cap = cv2.VideoCapture(video_path)  # read video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # calc video width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # calc video height
fps = int(cap.get(cv2.CAP_PROP_FPS))  # calc video frame/sec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # video codec setting
# Create VideoWriter object to save the output
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()  # read a frame from the video
    if not ret:  # if failed to read a frame,
        break  # exit the while loop

    results = model(frame)  # start object detection
    # get the object detection results
    detections = results.pred[0].cpu().numpy()
    bbox_xywh = []  # create a list to store bounding box information for objects, which will be used by the DeepSORT algorithm
    confidences = []  # create a list to store confidence scores for objects, which will be used for object tracking and visualization
    class_idxs = []  # create a list to store class indices for objects, which will be used for object tracking and visualization

    # initialize path_image (if it has not been initialized yet)
    if path_image is None:
        # create an image with the same size as the frame
        path_image = np.zeros_like(frame)

    # add information for each detected object to the corresponding list
    for x1, y1, x2, y2, conf, cls in detections:
        # calculate the center coordinates, width, and height of the bounding box
        bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        # store the confidence score for the object
        confidences.append(float(conf))
        # store the class index for the object (e.g., person, car, motorcycle, etc.)
        class_idxs.append(int(cls))

    # if there are any detected objects:
    if len(bbox_xywh) > 0:
        # convert the bounding boxes and confidence to numpy arrays
        bbox_xywh = np.array(bbox_xywh)
        confidences = np.array(confidences)
        # update the trackers using the DeepSORT algorithm
        trackers = deepsort.update(bbox_xywh, confidences, frame)
    else:
        trackers = []  # return an empty list if no objects are detected

    # save the paths of all detected objects in the trackers
    for t in trackers:
        # store all the information in the tracker in a variable (obj_id) to track the object's path
        obj_id = int(t[4])
        if obj_id not in object_paths.keys():
            # initialize the object's path dictionary
            object_paths[obj_id] = []

    # Visualization
    # Iterate over all detected objects
    for idx, track in enumerate(trackers[:len(class_idxs)]):
        x1, y1, x2, y2, track_id = track  # Extract tracker coordinates and object ID
        class_idx = class_idxs[idx]  # Extract object class index
        # Use class index to get class name
        class_id = results.names[class_idx]
        color = get_color(class_idx)  # Get color information for the object

        # If the object was detected in the previous frame as well:
        if track_id in prev_trackers:
            # prev_x1, prev_x2, prev_y1, prev_y2 are the coordinates of the object in the previous frame
            # x1, x2, y1, y2 are the coordinates of the object in the current frame
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_trackers[track_id]
            # Connect the center coordinates of the object in the previous and current frames with a line
            cv2.line(frame, (int((prev_x1 + prev_x2) / 2), int((prev_y1 + prev_y2) / 2)),
                     (int((x1 + x2) / 2), int((y1 + y2) / 2)), color, 2)

        # Draw bounding box and object ID on the current frame
        cv2.rectangle(frame, (int(x1), int(y1)),  # start coordinates: (x1, y1)
                      (int(x2), int(y2)), color, 2)  # finish coordinates: (x2, y2), color, thickness
        cv2.putText(frame, f"{class_id}-{track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # order : text, location, font, size, color, thickness

        # Update object trajectory
        if track_id not in trajectories:  # If the object ID is not in trajectories yet:
            # Collect the center coordinates of the object and create a new trajectory
            trajectories[track_id] = [(int((x1+x2)/2), int((y1+y2)/2))]
        else:  # If it is already in trajectories:
            # Add the center coordinates of the object to the existing trajectory
            trajectories[track_id].append((int((x1+x2)/2), int((y1+y2)/2)))

        # Draw the object trajectory
        # If the length of the trajectory is greater than 1:
        if len(trajectories[track_id]) > 1:
            # Iterate over all points in the trajectory and draw lines between them
            for i in range(1, len(trajectories[track_id])):
                cv2.line(frame, trajectories[track_id][i-1],
                         trajectories[track_id][i], (0, 255, 0), 2)  # prev_coordi, cur_coordi, color, thickness

    # Save current trackers as previous trackers
    prev_trackers = {track[-1]: track[:-1] for track in trackers}
    # Combine path image and current frame
    combined_frame = cv2.addWeighted(frame, 0.8, path_image, 0.2, 0)

    cv2.imshow("Result", frame)  # Display the output
    out.write(frame)  # Store the output

    # press 'q' to quite
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
out.release()  # video writer release
cap.release()  # video capture release
cv2.destroyAllWindows()  # OpenCV window close
