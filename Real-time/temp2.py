import torch
import cv2
import os
import numpy as np
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import tkinter as tk
from tkinter import filedialog

# YOLOv5 모델 불러오기 yolov5x버전
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# 딥소트 가져와서 병합
cfg = get_config()
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(current_dir, "deep_sort.yaml")
cfg.merge_from_file(config_file)

checkpoint_path = os.path.join(
    current_dir, "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")

# 객세 색상 생성 함수


def get_color(idx):
    # 기본 난수 생성기 초기화
    np.random.default_rng()
    # 0~224 사이의 랜덤 정수 3개 생성 후 튜플로 반환
    return tuple(map(int, np.random.randint(0, 255, size=3)))


# 딥소트 설정
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

# 객체생성
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select a video file")

# Video Capture 객체 생성 :: 선택한 동영상 읽기
cap = cv2.VideoCapture(video_path)  # 동영상 읽기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 동영상 가로 계산
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 도영상 높이 계산
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 동영상 초당 프레임 계산
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 동영상 파일 코덱 설정
# 결과물 저장을 위한 VideoWriter 객체 생성
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

prev_trackers = {}  # 트레커(경로추척)를 위한 딕셔너리
object_paths = {}  # 객체의 이동 경로를 추적하기 위한 dictionary
prev_frame_trackers = {}  # 이전 프레임의 경로 추적 딕셔너리
path_image = None  # 경로 이미지 저장위해 일단 초기화
trajectories = {}  # 객체의 궤적 저장 딕셔너리

while cap.isOpened():
    ret, frame = cap.read()  # 동영상의 프레임 읽기
    if not ret:  # 프레임 읽기 실패시
        break  # while문 종료

    results = model(frame)  # 객체감지 시작
    detections = results.pred[0].cpu().numpy()  # 객체감지 결과 가져오기
    bbox_xywh = []  # 객체의 바운딩 박스 저장 리스트 생성 ==> 나중에 DeepSort 알고리즘에서 사용됨
    confidences = []  # 객체의 신뢰도(확률) 저장 리스트     ==> 나중에 객체 추적 및 시각화에 사용
    class_idxs = []  # 객체의 클래스 인덱스 저장 리스트 생성  ==> 나중에 객체 추적 및 시각화에 사용

    # 경로 이미지 초기화 (초기화가 안되어있다면)
    if path_image is None:
        path_image = np.zeros_like(frame)  # 동영상의 프레임과 동일한 크기의 이미지 생성

    # 감지된 객체들의 정보를 각각의 리스트에 추가
    for x1, y1, x2, y2, conf, cls in detections:  # 감지 결과:
        # 바운딩 박스의 중심 좌표와 너비, 높이 계산
        # x1, y1 : 바운딩 박스의 왼쪽 위
        # x2, y2 : 바운딩 박스의 오른쪽 아래
        # 아래코드는 (x1 + x2) / 2: 박스의 가로중심좌표 && (y1 + y2) / 2: 박스의 세로중심좌표
        # x2 - x1 : 박스의 너비 && y2 - y1: 박스의 높이
        bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        confidences.append(float(conf))  # 객체의 신뢰도 저장
        class_idxs.append(int(cls))  # 클래스 인덱스 저장(사람, 자동차, 오토바이 등...)

    # 감지된 객체가 있다면 :
    if len(bbox_xywh) > 0:
        # 바운딩 박스와 신뢰도 배열을 numpy배열로 변환
        bbox_xywh = np.array(bbox_xywh)
        confidences = np.array(confidences)
        # deepsort를 사용하여 트레커 업데이트 => by using DeepSORT 알고리즘
        trackers = deepsort.update(bbox_xywh, confidences, frame)

    else:
        trackers = []  # 감지된 객체가 없으면 빈 리스트를 반환

    # 트레커에 감지된 모든 객체의 경로 저장
    for t in trackers:
        obj_id = int(t[4])  # 트래커에 저장된 모든 정보를 변수(obj_id)에 저장 -->> 객체의 이동경로 추적가능
        if obj_id not in object_paths.keys():
            object_paths[obj_id] = []  # 객채 경로 딕셔너리 초기화시키기

    # 시각화 준비 && 시작
    for idx, track in enumerate(trackers):  # track에 저장된 모든 정보들 추출시작
        x1, y1, x2, y2, track_id = track  # 트래커 좌표와 객체의 id 추출
        class_idx = class_idxs[idx]  # 객체의 클래스 인덱스 추출
        class_id = results.names[class_idx]  # 클래스 인덱스를 사용해서 클래스 이름 추출
        color = get_color(class_idx)  # 객체의 색상정보

        if track_id in prev_trackers:  # 이전에 감지된 객체가 현재 프레임에도 감지된다면:
            # prev_x1, prev_x2, prev_y1, prev_y2는 이전에 감지된 객체의 좌표
            # x1, x2, y1, y2는 현재 프레임에서 감지된 객체의 좌표
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_trackers[track_id]
            cv2.line(frame, (int((prev_x1 + prev_x2) / 2), int((prev_y1 + prev_y2) / 2)),
                     (int((x1 + x2) / 2), int((y1 + y2) / 2)), color, 2)  # 이전 프레임과 현재 프레임의 중심좌표 잇기

        # 현재 프레임에 바운딩박스, 객체 id 그리기
        cv2.rectangle(frame, (int(x1), int(y1)),  # 시작좌표: (x1, y1)
                      (int(x2), int(y2)), color, 2)  # 끝 좌표: (x2, y2), 색상, 선 두께
        cv2.putText(frame, f"{class_id}-{track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 텍스트, 위치, 폰트, 크기, 색상, 선 두께 -- 차례대로

        # 현재 객체의 이동경로 업데이트
        if track_id not in trajectories:  # track_id가 trajectories에 없으면:
            # 객체의 중심좌표 수집후 새로운 trajectory 생성
            trajectories[track_id] = [(int((x1+x2)/2), int((y1+y2)/2))]
        else:  # 존재하면:
            # 객체의 중심좌표를 기존 trajectory에 추가
            trajectories[track_id].append((int((x1+x2)/2), int((y1+y2)/2)))

        # 이동경로 그리기
        if len(trajectories[track_id]) > 1:  # 만약 trajectories의 길이가 2개 이상이면,
            for i in range(1, len(trajectories[track_id])):  # 모든 점을 돌면서 선을 그림
                cv2.line(frame, trajectories[track_id][i-1],
                         trajectories[track_id][i], (0, 255, 0), 2)  # 이전좌표, 현재좌표, 색상, 선 두께

    # 현재 프레임의 트래커를 이전 프레임의 트레커로 저장
    prev_trackers = {track[-1]: track[:-1] for track in trackers}
    # 경로 이미지랑 현재 프레임 합성하기
    combined_frame = cv2.addWeighted(frame, 0.8, path_image, 0.2, 0)

    cv2.imshow("Result", frame)  # 결과 출력
    out.write(frame)  # 결과 저장

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 결과 추출
out.release()  # 동영상 작성 종료
cap.release()  # 동영상 캡쳐 종료
cv2.destroyAllWindows()  # OpenCV 종료
