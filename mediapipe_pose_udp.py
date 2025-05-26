import cv2
import mediapipe as mp
import socket
import json
import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_args():
    parser = argparse.ArgumentParser(description='MediaPipe Pose UDP Broadcaster')
    parser.add_argument('--source', type=str, default='0', help='입력 소스: 0(웹캠), 파일 경로(비디오/이미지)')
    parser.add_argument('--udp_ip', type=str, default='255.255.255.255', help='UDP 브로드캐스트 IP')
    parser.add_argument('--udp_port', type=int, default=54321, help='UDP 포트')
    parser.add_argument('--log', type=str, default='', help='로그 파일 경로(선택)')
    return parser.parse_args()

def get_video_capture(source):
    # source가 숫자면 웹캠, 아니면 파일
    try:
        cam_idx = int(source)
        return cv2.VideoCapture(cam_idx)
    except ValueError:
        # 이미지 파일이면 반복 재생
        if os.path.splitext(source)[1].lower() in ['.jpg', '.png', '.jpeg']:
            img = cv2.imread(source)
            return img
        return cv2.VideoCapture(source)

# MediaPipe Pose 33개 본 이름 (Unreal 본 구조에 맞게 수정 가능)
POSE_BONE_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

# 부모 인덱스 (Unreal 본 계층에 맞게 수정 가능)
POSE_BONE_PARENTS = [
    -1,  # nose
    0, 1, 2, 0, 4, 5,  # eyes
    0, 0, 0, 0,        # ears, mouth
    0, 0, 11, 12,      # shoulders, elbows
    13, 14, 15, 16,    # wrists, pinky
    15, 16, 15, 16,    # index, thumb
    0, 0, 23, 24,      # hips, knees
    25, 26, 27, 28,    # ankles, heels
    29, 30             # foot_index
]

def get_bone_rotation(parent_pos, child_pos):
    v = np.array(child_pos) - np.array(parent_pos)
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return [0,0,0,1]
    v_norm = v / norm
    rot = R.align_vectors([v_norm], [[1,0,0]])[0].as_quat()
    return rot.tolist()  # [x, y, z, w]

# 스켈레톤 메시지 전송 함수
def send_subject(sock, UDP_IP, UDP_PORT, log_file=None):
    subject_message = {
        "mycharacter4": [
            {"Type": "CharacterSubject"},
        ] + [
            {"Name": name, "Parent": str(POSE_BONE_PARENTS[i]) if POSE_BONE_PARENTS[i] != -1 else "-1"}
            for i, name in enumerate(POSE_BONE_NAMES)
        ]
    }
    msg = json.dumps(subject_message).encode('utf-8')
    sock.sendto(msg, (UDP_IP, UDP_PORT))
    if log_file:
        with open(log_file, 'a') as f:
            f.write(msg.decode('utf-8') + '\n')

# 프레임별 애니메이션 메시지 전송 함수
def send_frame(sock, UDP_IP, UDP_PORT, frame_idx, landmarks, log_file=None):
    # landmarks: mediapipe 33개 [[x, y, z, visibility], ...]
    bone_transforms = []
    for i, lm in enumerate(landmarks):
        loc = [lm[0], lm[1], lm[2]]
        parent_idx = POSE_BONE_PARENTS[i]
        if parent_idx == -1 or parent_idx >= len(landmarks):
            rot = [0,0,0,1]
        else:
            rot = get_bone_rotation(landmarks[parent_idx][:3], loc)
        bone_transforms.append({
            "Location": loc,
            "Rotation": rot,
            "Scale": [1,1,1]
        })
    animation_message = {
        "frame": frame_idx,
        "mycharacter4": [
            {"Type": "CharacterAnimation"},
            *bone_transforms
        ]
    }
    msg = json.dumps(animation_message).encode('utf-8')
    sock.sendto(msg, (UDP_IP, UDP_PORT))
    if log_file:
        with open(log_file, 'a') as f:
            f.write(msg.decode('utf-8') + '\n')

def main():
    args = parse_args()
    UDP_IP = args.udp_ip
    UDP_PORT = args.udp_port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    log_file = args.log if args.log else None

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = get_video_capture(args.source)
    is_image = isinstance(cap, (np.ndarray,))
    frame_idx = 0
    subject_sent = False

    while True:
        if is_image:
            frame = cap.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            landmark_list = []
            for lm in landmarks:
                landmark_list.append([lm.x, lm.y, lm.z, lm.visibility])
            if not subject_sent:
                send_subject(sock, UDP_IP, UDP_PORT, log_file)
                subject_sent = True
            send_frame(sock, UDP_IP, UDP_PORT, frame_idx, landmark_list, log_file)
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Pose', frame)
        frame_idx += 1
        key = cv2.waitKey(1)
        if key == 27:
            break
        if is_image:
            # 이미지 파일이면 한 번만 처리
            break
    if not is_image:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 