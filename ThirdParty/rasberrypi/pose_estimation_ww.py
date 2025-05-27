import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import hailo
from hailo_rpi_common import (
    get_default_parser,
    QUEUE,
    get_caps_from_pad,
    get_numpy_from_buffer,
    GStreamerApp,
    app_callback_class,
)
import socket
import json
from scipy.spatial.transform import Rotation as R

# UDP 설정 (필요시 수정)
UDP_IP = "192.168.0.255"  # 브로드캐스트 주소
UDP_PORT = 54321
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
skeleton_sent = False  # 스켈레톤 메시지 전송 여부 플래그

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_udp_log.txt')

def log_udp_message(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(msg.decode('utf-8') + '\n')

# 스켈레톤 구조 메시지 전송 함수
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

# 프레임별 애니메이션 메시지 전송 함수
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
    """
    부모-자식 본 사이의 회전을 계산합니다.
    target_axis: 본이 가리키는 방향 (기본값: x축)
    """
    v = np.array(child_pos) - np.array(parent_pos)
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return [0,0,0,1]
    v_norm = v / norm
    rot = R.align_vectors([v_norm], [target_axis])[0].as_quat()
    return rot.tolist()  # [x, y, z, w]

def map_2d_to_3d(
    keypoints_2d: np.ndarray,
    scale: float,
    floor_angle_deg: float,
    origin_px: tuple = (0, 0),
    depth: float = 0.0,
    to_unreal: bool = False,
    unreal_scale: float = 100.0
) -> np.ndarray:
    """
    2D COCO 키포인트(pixel) 배열을 3D 평면 좌표로 매핑합니다.
    """
    theta = np.deg2rad(floor_angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    pts = keypoints_2d - np.array(origin_px)
    x_plane = pts[:, 0] * c + pts[:, 1] * s
    y_plane = -pts[:, 0] * s + pts[:, 1] * c
    x_world = x_plane * scale
    y_world = y_plane * scale
    z_world = np.full_like(x_world, depth)
    points3d = np.stack((x_world, y_world, z_world), axis=1)
    if to_unreal:
        points3d *= unreal_scale
    return points3d

def detect_floor_angle(points_2d: np.ndarray) -> float:
    """
    바닥 각도를 자동으로 감지합니다.
    """
    left_hip = points_2d[11]
    right_hip = points_2d[12]
    left_ankle = points_2d[15]
    right_ankle = points_2d[16]
    hip_mid = (left_hip + right_hip) / 2
    ankle_mid = (left_ankle + right_ankle) / 2
    dx = ankle_mid[0] - hip_mid[0]
    dy = ankle_mid[1] - hip_mid[1]
    angle_rad = np.arctan2(dy, dx)
    return np.rad2deg(angle_rad)

def calculate_scale(points_2d: np.ndarray, real_height_m: float = 1.7) -> float:
    """
    실제 키를 기반으로 픽셀-미터 스케일을 계산합니다.
    """
    head = points_2d[0]
    left_ankle = points_2d[15]
    right_ankle = points_2d[16]
    ankle_mid = (left_ankle + right_ankle) / 2
    pixel_height = np.linalg.norm(head - ankle_mid)
    return real_height_m / pixel_height

def get_bone_rotations(points_3d):
    """
    스켈레톤 구조에 맞는 13개의 본 회전을 계산합니다.
    """
    rotations = []
    # 스켈레톤 구조에 맞는 본 매핑 (COCO 키포인트 -> 스켈레톤 본)
    bone_mapping = {
        'head': 0,           # nose
        'upperarm_l': 11,    # left_shoulder
        'upperarm_r': 12,    # right_shoulder
        'lowerarm_l': 13,    # left_elbow
        'lowerarm_r': 14,    # right_elbow
        'hand_l': 15,        # left_wrist
        'hand_r': 16,        # right_wrist
        'thigh_l': 23,       # left_hip
        'thigh_r': 24,       # right_hip
        'calf_l': 25,        # left_knee
        'calf_r': 26,        # right_knee
        'foot_l': 27,        # left_ankle
        'foot_r': 28,        # right_ankle
    }
    
    # 본 계층 구조 정의
    bone_hierarchy = [
        ('head', None),          # head는 부모 없음
        ('upperarm_l', 'head'),  # upperarm_l의 부모는 head
        ('upperarm_r', 'head'),  # upperarm_r의 부모는 head
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
    
    # 각 본의 회전 계산
    for bone_name, parent_name in bone_hierarchy:
        if parent_name is None:
            # head는 기본 회전
            rotations.append([0,0,0,1])
        else:
            # 부모-자식 본 사이의 회전 계산
            parent_idx = bone_mapping[parent_name]
            child_idx = bone_mapping[bone_name]
            if parent_idx < len(points_3d) and child_idx < len(points_3d):
                rot = calculate_bone_rotation(points_3d[parent_idx], points_3d[child_idx])
                rotations.append(rot)
            else:
                rotations.append([0,0,0,1])
    
    return rotations

def get_bone_positions(points_3d):
    """
    스켈레톤 구조에 맞는 13개의 본 위치를 계산합니다.
    """
    positions = []
    # 스켈레톤 구조에 맞는 본 매핑
    bone_mapping = {
        'head': 0,           # nose
        'upperarm_l': 11,    # left_shoulder
        'upperarm_r': 12,    # right_shoulder
        'lowerarm_l': 13,    # left_elbow
        'lowerarm_r': 14,    # right_elbow
        'hand_l': 15,        # left_wrist
        'hand_r': 16,        # right_wrist
        'thigh_l': 23,       # left_hip
        'thigh_r': 24,       # right_hip
        'calf_l': 25,        # left_knee
        'calf_r': 26,        # right_knee
        'foot_l': 27,        # left_ankle
        'foot_r': 28,        # right_ankle
    }
    
    # 각 본의 위치 추출
    for bone_name in ['head', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 'lowerarm_r',
                     'hand_l', 'hand_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r',
                     'foot_l', 'foot_r']:
        idx = bone_mapping[bone_name]
        if idx < len(points_3d):
            positions.append(points_3d[idx].tolist())
        else:
            positions.append([0,0,0])
    
    return positions

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    global skeleton_sent
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
        
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"
    
    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            string_to_print += (f"Detection: {label} {confidence:.2f}\n")
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                
                # 2D 포인트 추출
                points_2d = np.array([[p.x() * width, p.y() * height] for p in points])
                
                # 바닥 각도 감지
                floor_angle = detect_floor_angle(points_2d)
                
                # 스케일 계산
                scale = calculate_scale(points_2d)
                
                # 3D 좌표로 변환
                points_3d = map_2d_to_3d(
                    keypoints_2d=points_2d,
                    scale=scale,
                    floor_angle_deg=floor_angle,
                    origin_px=(width/2, height/2),
                    depth=0.0,
                    to_unreal=True
                )
                
                if not skeleton_sent:
                    send_skeleton_structure()
                    skeleton_sent = True
                
                # 본 회전과 위치 계산
                bone_rotations = get_bone_rotations(points_3d)
                bone_positions = get_bone_positions(points_3d)
                
                # 본 트랜스폼 생성 (13개 본에 맞춤)
                bone_transforms = []
                for i in range(13):  # 13개의 본
                    bone_transforms.append({
                        "Location": bone_positions[i],
                        "Rotation": bone_rotations[i],
                        "Scale": [1,1,1]
                    })
                
                send_frame_animation(bone_transforms)
                
                if user_data.use_frame:    
                    for point_2d in points_2d:
                        cv2.circle(frame, (int(point_2d[0]), int(point_2d[1])), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"Floor Angle: {floor_angle:.1f}°", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if user_data.use_frame:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------
class GStreamerPoseEstimationApp(GStreamerApp):
    def __init__(self, args, user_data):
        super().__init__(args, user_data)
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        self.default_postprocess_so = os.path.join(self.postprocess_dir, 'libyolov8pose_post.so')
        self.post_function_name = "filter"
        self.hef_path = os.path.join(self.current_path, 'yolov8s_pose_h8l_pi.hef')
        self.app_callback = app_callback
        setproctitle.setproctitle("Hailo Pose Estimation App")
        self.create_pipeline()

    def get_pipeline_string(self):
        if (self.source_type == "rpi"):
            source_element = f"libcamerasrc name=src_0 auto-focus-mode=2 ! "
            source_element += f"video/x-raw, format={self.network_format}, width=1536, height=864 ! "
            source_element += QUEUE("queue_src_scale")
            source_element += f"videoscale ! "
            source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, framerate=30/1 ! "
        
        elif (self.source_type == "usb"):
            source_element = f"v4l2src device={self.video_source} name=src_0 ! "
            source_element += f"video/x-raw, width=640, height=480, framerate=30/1 ! "
        else:  
            source_element = f"filesrc location={self.video_source} name=src_0 ! "
            source_element += QUEUE("queue_dec264")
            source_element += f" qtdemux ! h264parse ! avdec_h264 max-threads=2 ! "
            source_element += f" video/x-raw,format=I420 ! "
        source_element += QUEUE("queue_scale")
        source_element += f"videoscale n-threads=2 ! "
        source_element += QUEUE("queue_src_convert")
        source_element += f"videoconvert n-threads=3 name=src_convert qos=false ! "
        source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, pixel-aspect-ratio=1/1 ! "
        
        pipeline_string = "hailomuxer name=hmux "
        pipeline_string += source_element
        pipeline_string += "tee name=t ! "
        pipeline_string += QUEUE("bypass_queue", max_size_buffers=20) + "hmux.sink_0 "
        pipeline_string += "t. ! " + QUEUE("queue_hailonet")
        pipeline_string += "videoconvert n-threads=3 ! "
        pipeline_string += f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} force-writable=true ! "
        pipeline_string += QUEUE("queue_hailofilter")
        pipeline_string += f"hailofilter function-name={self.post_function_name} so-path={self.default_postprocess_so} qos=false ! "
        pipeline_string += QUEUE("queue_hmuc") + " hmux.sink_1 "
        pipeline_string += "hmux. ! " + QUEUE("queue_hailo_python")
        pipeline_string += QUEUE("queue_user_callback")
        pipeline_string += f"identity name=identity_callback ! "
        pipeline_string += QUEUE("queue_hailooverlay")
        pipeline_string += f"hailooverlay ! "
        pipeline_string += QUEUE("queue_videoconvert")
        pipeline_string += f"videoconvert n-threads=3 qos=false ! "
        pipeline_string += QUEUE("queue_hailo_display")
        pipeline_string += f"fpsdisplaysink video-sink={self.video_sink} name=hailo_display sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true "
        print(pipeline_string)
        return pipeline_string
    
if __name__ == "__main__":
    user_data = user_app_callback_class()
    parser = get_default_parser()
    args = parser.parse_args()
    app = GStreamerPoseEstimationApp(args, user_data)
    app.run()