import cv2
import math
import numpy as np
import depthai as dai
from ultralytics import YOLO
from time import sleep
import threading
from codrone_edu.drone import Drone
from PIL import Image
import torch
import queue

# 모델 로드
pose_model = YOLO('yolov8n-pose.pt')
drone_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

DRONE_CLASS_NAME = 'drone'

# 드론 초기화
drone = Drone()
drone.pair()
drone.set_trim(0, 0)

# 상태 변수
drone_flying = False
fall_triggered = False

# 프레임 전달용 큐
frame_queue = queue.Queue(maxsize=1)

# 사람과의 거리 판단
def is_close_enough(bbox, threshold=200):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    return h >= threshold

# 사람에게 접근 후 착륙
def approach_and_land():
    global drone_flying, fall_triggered

    while True:
        try:
            frame2 = frame_queue.get(timeout=2)
        except queue.Empty:
            continue

        img = Image.fromarray(frame2[..., ::-1])
        results = drone_model(img, size=640)

        frame_width = frame2.shape[1]
        frame_center = frame_width // 2
        tolerance = 40

        person_found = False
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result.tolist()
            if conf > 0.5 and int(cls) == 0:
                person_found = True
                person_center_x = int((x1 + x2) / 2)

                if not drone_flying:
                    drone.takeoff()
                    drone_flying = True
                    sleep(1)

                if person_center_x < frame_center - tolerance:
                    drone.set_roll(-30)
                    drone.move(1)
                    drone.set_roll(0)
                elif person_center_x > frame_center + tolerance:
                    drone.set_roll(30)
                    drone.move(1)
                    drone.set_roll(0)
                else:
                    if is_close_enough((x1, y1, x2, y2)):
                        drone.land()
                        drone_flying = False
                        fall_triggered = False
                        return
                    drone.set_pitch(30)
                    drone.move(1)
                    drone.set_pitch(0)
                break

        if not person_found:
            drone.set_pitch(30)
            drone.move(1)
            drone.set_pitch(0)

        sleep(0.3)

# 낙상 감지용 변수들
position_history = {}
angle_history = {}
last_valid = {}

velocity_threshold = 20
angle_change_threshold = 45
aspect_ratio_threshold = 1.5
conf_threshold = 0.3
pt_radius = 3
ln_thick = 1

bright_colors = {
    'nose_neck': (200,255,255), 'neck_l_sh': (255,200,255), 'neck_r_sh': (255,255,200),
    'neck_l_hip': (200,230,255), 'neck_r_hip': (255,225,200),
    'lwrist_lelb': (230,255,200), 'lelb_lsh': (255,200,230),
    'rwrist_relb': (200,255,230), 'relb_rsh': (230,200,255),
    'lhip_lkne': (255,240,200), 'lkne_lank': (200,255,240),
    'rhip_rkne': (240,200,255), 'rkne_rank': (255,200,200)
}

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

# 카메라 초기화
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
        if not ret: break
    else:
        frame = q.get().getCvFrame()

    frame = cv2.flip(frame, 1)

    # 드론 검출
    img_pil = Image.fromarray(frame[..., ::-1])
    drone_results = drone_model(img_pil, size=640)
    drone_detected = False
    for res in drone_results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = res.tolist()
        if conf > 0.5 and drone_model.names[int(cls)].lower() == DRONE_CLASS_NAME:
            drone_detected = True
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 3)
            cv2.putText(frame, "Drone", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

    # 시야에 드론이 없으면 전진
    if not drone_detected:
        drone.set_pitch(30)
        drone.move(1)
        drone.set_pitch(0)

    # 포즈 검출 및 낙상 감지
    results = pose_model(frame, verbose=False)
    for r in results:
        for box, kps in zip(r.boxes, r.keypoints.data):
            if int(box.cls[0]) != 0:
                continue
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            w, h = x2-x1, y2-y1
            center_y = (y1+y2)//2
            angle = math.degrees(math.atan2(h, w))
            ar = w / (h or 1)
            pid = f"{x1}_{y1}_{x2}_{y2}"

            position_history.setdefault(pid, []).append(center_y)
            angle_history.setdefault(pid, []).append(angle)
            position_history[pid] = position_history[pid][-2:]
            angle_history[pid] = angle_history[pid][-2:]

            vel = (position_history[pid][1] - position_history[pid][0]) if len(position_history[pid])==2 else 0
            ang_chg = abs(angle_history[pid][1] - angle_history[pid][0]) if len(angle_history[pid])==2 else 0
            fall = ar > aspect_ratio_threshold or vel > velocity_threshold or ang_chg > angle_change_threshold

            if fall and not fall_triggered:
                print("Fall detected, drone taking off!")
                fall_triggered = True
                if not drone_flying:
                    drone.takeoff()
                    drone_flying = True
                    sleep(1)
                threading.Thread(target=approach_and_land, daemon=True).start()

            if fall_triggered:
                if frame_queue.full():
                    try: frame_queue.get_nowait()
                    except queue.Empty: pass
                frame_queue.put(frame)

            pose = kps.cpu().numpy() if hasattr(kps,'cpu') else np.array(kps)
            idxs = [0,5,6,7,8,9,10,11,12,13,14,15,16]
            pts = {}
            for i in idxs:
                x,y,c = pose[i]
                pts[i] = (x,y) if c>=conf_threshold else None
            last_valid.setdefault(pid,{})
            for i in idxs:
                if pts[i] is None and i in last_valid[pid]:
                    pts[i] = last_valid[pid][i]

            neck = None
            if is_valid(pts.get(5)) and is_valid(pts.get(6)):
                neck = ((pts[5][0]+pts[6][0])/2, (pts[5][1]+pts[6][1])/2)

            for p1,p2,key in connections:
                pt1 = neck if p1=='neck' else pts.get(p1)
                pt2 = neck if p2=='neck' else pts.get(p2)
                if is_valid(pt1) and is_valid(pt2):
                    color = (0,0,255) if fall else bright_colors[key]
                    cv2.line(frame, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), color, ln_thick)

            for i in idxs:
                p = neck if i=='neck' else pts.get(i)
                if is_valid(p):
                    color = (0,0,255) if fall else (255,255,0)
                    cv2.circle(frame, (int(p[0]),int(p[1])), pt_radius, color, -1)

            col = (0,0,255) if fall else (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            if fall:
                cv2.putText(frame, "Fall Detected", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)

            last_valid[pid] = pts

    cv2.imshow("Fall Detection with Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

if use_webcam:
    cap.release()
else:
    device.close()
cv2.destroyAllWindows()
