import cv2
import socket
import json
import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import hailo

def parse_args():
    parser = argparse.ArgumentParser(description='Hailo Pose UDP Broadcaster')
    parser.add_argument('--udp_ip', type=str, default='255.255.255.255', help='UDP 브로드캐스트 IP')
    parser.add_argument('--udp_port', type=int, default=54321, help='UDP 포트')
    parser.add_argument('--log', type=str, default='', help='로그 파일 경로(선택)')
    parser.add_argument('--hef', type=str, default='Resources/hailo8/pose_landmark_full.hef', help='HEF 파일 경로')
    parser.add_argument('--camera', type=int, default=0, help='카메라 인덱스')
    return parser.parse_args()

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

def send_frame(sock, UDP_IP, UDP_PORT, frame_idx, landmarks, log_file=None):
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
    hef_path = args.hef
    cam_idx = args.camera

    device = hailo.Device()
    network_group = device.create_hef(hef_path)
    input_vstream_info = network_group.get_input_vstream_infos()[0]
    output_vstream_info = network_group.get_output_vstream_infos()[0]

    with hailo.vstream.InputVStreams(network_group, [input_vstream_info]) as input_vstreams, \
         hailo.vstream.OutputVStreams(network_group, [output_vstream_info]) as output_vstreams:
        cap = cv2.VideoCapture(cam_idx)
        frame_idx = 0
        subject_sent = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 전처리: 네트워크 입력 크기에 맞게 resize, RGB 변환 등
            input_frame = cv2.resize(frame, (input_vstream_info.shape[2], input_vstream_info.shape[1]))
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            input_frame = np.expand_dims(input_frame, axis=0)
            # 추론
            input_vstreams[0].send(input_frame)
            output = output_vstreams[0].recv()
            # output: (1, 33, 4) 등 mediapipe와 동일한 구조라고 가정
            landmark_list = output[0].tolist()  # [[x, y, z, visibility], ...]
            if not subject_sent:
                send_subject(sock, UDP_IP, UDP_PORT, log_file)
                subject_sent = True
            send_frame(sock, UDP_IP, UDP_PORT, frame_idx, landmark_list, log_file)
            frame_idx += 1
            cv2.imshow('Hailo Pose', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 