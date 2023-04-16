import torch
import cv2
from yolov5 import models
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import tkinter as tk
from tkinter import filedialog

model = models.yolov5s(pretrained=True)

cfg = get_config()
cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

deepsort = DeepSort(
    cfg.DEEPSORT.REID_CKPT,
    max_dist=cfg.DEEPSORT.MAX_DIST,
    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg.DEEPSORT.MAX_AGE,
    n_init=cfg.DEEPSORT.N_INIT,
    nn_budget=cfg.DEEPSORT.NN_BUDGET,
    use_cuda=True
)

# 사용자에게 동영상 파일 선택 요청
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select a video file")

cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 탐지 수행
    results = model(frame)

    # 객체 추적 수행
    detections = results.xyxy[0].numpy()
    trackers = deepsort.update(detections)

    # 시각화
    for track in trackers:
        bbox = track.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
            bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.putText(frame, f"{track.get_class()}-{track.track_id}", (int(bbox[0]), int(
            bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Result", frame)

    # 키 입력을 대기하고 'q' 키를 누르면 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
