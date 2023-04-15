from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from yolov5.utils.datasets import LoadImages
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
import sys
from pathlib import Path
import cv2

# DeepSORT와 YOLOv5 디렉토리를 PATH에 추가
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR / "yolov5"))
sys.path.append(str(CURRENT_DIR / "Yolov5_DeepSort_Pytorch"))


def main(input_video, output_video, yolo_weights):
    device = select_device('')

    # YOLOv5 모델 로드
    model = attempt_load(yolo_weights, map_location=device)
    img_size = check_img_size(640, s=model.stride.max())

    # DeepSORT 설정
    cfg = get_config()
    cfg.merge_from_file("Yolov5_DeepSort_Pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE,
                        n_init=cfg.DEEPSORT.N_INIT,
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # 동영상 처리
    dataset = LoadImages(input_video, img_size=img_size)
    writer = None

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16
