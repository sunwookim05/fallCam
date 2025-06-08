import cv2
import math
import numpy as np
import depthai as dai
from ultralytics import YOLO
from time import sleep
from codrone_edu.drone import *

# 모델 로드
model = YOLO('yolov8n-pose.pt')

# 초기값
position_history = {}
angle_history = {}
velocity_threshold = 20
angle_change_threshold = 45
aspect_ratio_threshold = 1.5
pt_radius = 3
ln_thick = 1
conf_threshold = 0.3
last_valid = {}

# 색상 팔레트
bright_colors = {
    'nose_neck': (200, 255, 255), 'neck_l_sh': (255, 200, 255), 'neck_r_sh': (255, 255, 200),
    'neck_l_hip': (200, 230, 255), 'neck_r_hip': (255, 225, 200),
    'lwrist_lelb': (230, 255, 200), 'lelb_lsh': (255, 200, 230),
    'rwrist_relb': (200, 255, 230), 'relb_rsh': (230, 200, 255),
    'lhip_lkne': (255, 240, 200), 'lkne_lank': (200, 255, 240),
    'rhip_rkne': (240, 200, 255), 'rkne_rank': (255, 200, 200)
}

# 연결할 관절 쌍 정의
connections = [
    (0, 'neck', 'nose_neck'),
    ('neck', 5, 'neck_l_sh'), ('neck', 6, 'neck_r_sh'),
    ('neck', 11, 'neck_l_hip'), ('neck', 12, 'neck_r_hip'),
    (7, 9, 'lwrist_lelb'), (5, 7, 'lelb_lsh'),
    (8, 10, 'rwrist_relb'), (6, 8, 'relb_rsh'),
    (11, 13, 'lhip_lkne'), (13, 15, 'lkne_lank'),
    (12, 14, 'rhip_rkne'), (14, 16, 'rkne_rank')
]

def is_valid(pt):
    return pt is not None and np.isfinite(pt).all()

# --- OAK-D 또는 웹캠 선택 ---
use_webcam = False
try:
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam_rgb.video.link(xout.input)

    device = dai.Device(pipeline)
    q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
except:
    print("OAK-D not found, switching to default webcam.")
    use_webcam = True
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Fall Detection with Pose", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Fall Detection with Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    if use_webcam:
        ret, frame = cap.read()
        if not ret:
            break
    else:
        frame = q.get().getCvFrame()

    frame = cv2.flip(frame, 1)

    results = model(frame, verbose=False)
    for r in results:
        for box, kps in zip(r.boxes, r.keypoints.data):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if int(box.cls[0]) != 0:
                continue

            # 낙상 판정
            w, h = x2 - x1, y2 - y1
            center_y = (y1 + y2) // 2
            angle = math.degrees(math.atan2(h, w))
            ar = w / (h or 1)
            pid = f"{x1}_{y1}_{x2}_{y2}"
            position_history.setdefault(pid, []).append(center_y)   
            angle_history.setdefault(pid, []).append(angle)
            position_history[pid] = position_history[pid][-2:]
            angle_history[pid] = angle_history[pid][-2:]
            vel = (position_history[pid][1] - position_history[pid][0]) if len(position_history[pid]) == 2 else 0
            ang_chg = abs(angle_history[pid][1] - angle_history[pid][0]) if len(angle_history[pid]) == 2 else 0
            fall = ar > aspect_ratio_threshold or vel > velocity_threshold or ang_chg > angle_change_threshold

            # 낙상 텍스트
            if fall:
                cv2.putText(frame, "Fall Detected", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            # 키포인트 가져오기
            pose = kps.cpu().numpy() if hasattr(kps, 'cpu') else np.array(kps)
            idxs = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            pts = {}
            for i in idxs:
                x, y, c = pose[i]
                pts[i] = (x, y) if c >= conf_threshold else None
            last_valid.setdefault(pid, {})
            for i in idxs:
                if pts[i] is None and i in last_valid[pid]:
                    pts[i] = last_valid[pid][i]

            # 목(neck) 위치 계산
            neck = None
            if is_valid(pts.get(5)) and is_valid(pts.get(6)):
                neck = ((pts[5][0] + pts[6][0]) / 2, (pts[5][1] + pts[6][1]) / 2)

            # 선 그리기
            for p1, p2, key in connections:
                pt1 = neck if p1 == 'neck' else pts.get(p1)
                pt2 = neck if p2 == 'neck' else pts.get(p2)
                if is_valid(pt1) and is_valid(pt2):
                    color = (0, 0, 255) if fall else bright_colors[key]
                    cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, ln_thick)

            # 포인트 그리기
            for i in idxs:
                p = neck if i == 'neck' else pts.get(i)
                if is_valid(p):
                    color = (0, 0, 255) if fall else (255, 255, 0)
                    cv2.circle(frame, (int(p[0]), int(p[1])), pt_radius, color, -1)

            # 박스 그리기
            col = (0, 0, 255) if fall else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

            last_valid[pid] = pts

    cv2.imshow("Fall Detection with Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

if use_webcam:
    cap.release()
else:
    device.close()
cv2.destroyAllWindows()
