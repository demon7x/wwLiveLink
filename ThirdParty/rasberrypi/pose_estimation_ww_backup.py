import os
import argparse
import numpy as np
import time
import socket
import json
from scipy.spatial.transform import Rotation as R

# UDP 설정
UDP_IP = "192.168.0.255"  # 브로드캐스트 주소
UDP_PORT = 54321
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
skeleton_sent = False

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_udp_log.txt')

def log_udp_message(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(msg.decode('utf-8') + '\n')

def send_skeleton_structure():
    skeleton_message = {
        "mycharacter3": [
            {"Type": "CharacterSubject"},
            {"Name": "head", "Parent": "-1"},
            {"Name": "upperarm_l", "Parent": "1"},
            {"Name": "upperarm_r", "Parent": "1"},
            {"Name": "lowerarm_l", "Parent": "2"},
            {"Name": "lowerarm_r", "Parent": "3"},
            {"Name": "hand_l", "Parent": "4"},
            {"Name": "hand_r", "Parent": "5"},
            {"Name": "thigh_l", "Parent": "1"},
            {"Name": "thigh_r", "Parent": "1"},
            {"Name": "calf_l", "Parent": "8"},
            {"Name": "calf_r", "Parent": "9"},
            {"Name": "foot_l", "Parent": "10"},
            {"Name": "foot_r", "Parent": "11"},
        ]
    }
    msg = json.dumps(skeleton_message).encode('utf-8')
    sock.sendto(msg, (UDP_IP, UDP_PORT))
    log_udp_message(msg)

def send_frame_animation(bone_transforms):
    message = {
        "mycharacter3": [
            {"Type": "CharacterAnimation"},
            *bone_transforms
        ]
    }
    msg = json.dumps(message).encode('utf-8')
    sock.sendto(msg, (UDP_IP, UDP_PORT))
    log_udp_message(msg)

def calculate_bone_rotation(parent_pos, child_pos, target_axis=[1,0,0]):
    v = np.array(child_pos) - np.array(parent_pos)
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return [0,0,0,1]
    v_norm = v / norm
    rot = R.align_vectors([v_norm], [target_axis])[0].as_quat()
    return rot.tolist()

def read_trc_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 데이터 시작 라인 찾기
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('Frame#'):
            data_start = i + 2  # 헤더 다음 라인부터 데이터 시작
            break
    
    # 데이터 파싱
    data = []
    for line in lines[data_start:]:
        if line.strip():
            values = line.strip().split()
            try:
                frame_num = int(values[0])
                frame_data = {
                    'frame': frame_num,
                    'time': float(values[1]),
                    'points': []
                }
                # 3D 좌표 추출 (X, Y, Z)
                for i in range(2, len(values), 3):
                    if i + 2 < len(values):
                        x = float(values[i])
                        y = float(values[i+1])
                        z = float(values[i+2])
                        frame_data['points'].append([x, y, z])
                data.append(frame_data)
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid line: {line.strip()}")
                continue
    
    return data

def get_bone_rotations(points_3d):
    rotations = []
    # TRC 파일의 마커 구조에 맞는 본 매핑
    bone_mapping = {
        'head': 0,           # Hip
        'upperarm_l': 1,     # RHip
        'upperarm_r': 2,     # RKnee
        'lowerarm_l': 3,     # RAnkle
        'lowerarm_r': 4,     # RBigToe
        'hand_l': 5,         # RSmallToe
        'hand_r': 6,         # RHeel
        'thigh_l': 7,        # LHip
        'thigh_r': 8,        # LKnee
        'calf_l': 9,         # LAnkle
        'calf_r': 10,        # LBigToe
        'foot_l': 11,        # LSmallToe
        'foot_r': 12,        # LHeel
    }
    
    bone_hierarchy = [
        ('head', None),
        ('upperarm_l', 'head'),
        ('upperarm_r', 'head'),
        ('lowerarm_l', 'upperarm_l'),
        ('lowerarm_r', 'upperarm_r'),
        ('hand_l', 'lowerarm_l'),
        ('hand_r', 'lowerarm_r'),
        ('thigh_l', 'head'),
        ('thigh_r', 'head'),
        ('calf_l', 'thigh_l'),
        ('calf_r', 'thigh_r'),
        ('foot_l', 'calf_l'),
        ('foot_r', 'calf_r'),
    ]
    
    for bone_name, parent_name in bone_hierarchy:
        if parent_name is None:
            rotations.append([0,0,0,1])
        else:
            parent_idx = bone_mapping[parent_name]
            child_idx = bone_mapping[bone_name]
            if parent_idx < len(points_3d) and child_idx < len(points_3d):
                rot = calculate_bone_rotation(points_3d[parent_idx], points_3d[child_idx])
                rotations.append(rot)
            else:
                rotations.append([0,0,0,1])
    
    return rotations

def get_bone_positions(points_3d):
    positions = []
    # TRC 파일의 마커 구조에 맞는 본 매핑
    bone_mapping = {
        'head': 0,           # Hip
        'upperarm_l': 1,     # RHip
        'upperarm_r': 2,     # RKnee
        'lowerarm_l': 3,     # RAnkle
        'lowerarm_r': 4,     # RBigToe
        'hand_l': 5,         # RSmallToe
        'hand_r': 6,         # RHeel
        'thigh_l': 7,        # LHip
        'thigh_r': 8,        # LKnee
        'calf_l': 9,         # LAnkle
        'calf_r': 10,        # LBigToe
        'foot_l': 11,        # LSmallToe
        'foot_r': 12,        # LHeel
    }
    
    for bone_name in ['head', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 'lowerarm_r',
                     'hand_l', 'hand_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r',
                     'foot_l', 'foot_r']:
        idx = bone_mapping[bone_name]
        if idx < len(points_3d):
            positions.append(points_3d[idx].tolist())
        else:
            positions.append([0,0,0])
    
    return positions

def process_trc_file(file_path):
    global skeleton_sent
    
    # TRC 파일 읽기
    motion_data = read_trc_file(file_path)
    
    # 스켈레톤 구조 전송
    if not skeleton_sent:
        send_skeleton_structure()
        skeleton_sent = True
    
    # 각 프레임 처리
    for frame_data in motion_data:
        points_3d = np.array(frame_data['points'])
        
        # 본 회전과 위치 계산
        bone_rotations = get_bone_rotations(points_3d)
        bone_positions = get_bone_positions(points_3d)
        
        # 본 트랜스폼 생성
        bone_transforms = []
        for i in range(13):
            bone_transforms.append({
                "Location": bone_positions[i],
                "Rotation": bone_rotations[i],
                "Scale": [1,1,1]
            })
        
        # 프레임 데이터 전송
        send_frame_animation(bone_transforms)
        
        # 프레임 간 딜레이 (30fps 기준)
        time.sleep(1/30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trc_file', type=str, required=True, help='Path to TRC file')
    args = parser.parse_args()
    
    process_trc_file(args.trc_file)