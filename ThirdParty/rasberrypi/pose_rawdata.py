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

def map_2d_to_3d(
    keypoints_2d: np.ndarray,
    scale: float,
    floor_angle_deg: float,
    origin_px: tuple = (0, 0),
    depth: float = 0.0
) -> np.ndarray:
    theta = np.deg2rad(floor_angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    pts = keypoints_2d - np.array(origin_px)
    x_plane = pts[:, 0] * c + pts[:, 1] * s
    y_plane = -pts[:, 0] * s + pts[:, 1] * c
    x_world = x_plane * scale
    y_world = y_plane * scale
    z_world = np.full_like(x_world, depth)
    return np.stack((x_world, y_world, z_world), axis=1)

def quaternion_from_vectors(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    dot = np.dot(v0, v1)
    if dot < -0.999999:
        orth = np.array([1, 0, 0]) if abs(v0[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v0, orth)
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(v0, v1)
    s = np.sqrt((1.0 + dot) * 2.0)
    q_xyz = axis / s
    q_w = 0.5 * s
    q = np.concatenate([q_xyz, [q_w]])
    return q / np.linalg.norm(q)

def quaternion_to_euler(q: np.ndarray) -> tuple:
    x, y, z, w = q
    sinr = 2 * (w*x + y*z)
    cosr = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2 * (w*y - z*x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny = 2 * (w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny, cosy)
    return (np.rad2deg(pitch), np.rad2deg(yaw), np.rad2deg(roll))

def compute_bone_transforms_euler(
    keypoints_3d: dict,
    bone_map: dict,
    default_axis: np.ndarray = np.array([1.0, 0.0, 0.0]),
    scale_axis: str = 'identity'
) -> dict:
    transforms = {}
    for bone, (parent, child) in bone_map.items():
        p = np.array(keypoints_3d[parent], dtype=float)
        c = np.array(keypoints_3d[child], dtype=float)
        loc = ((p + c) / 2.0).tolist()
        dir_vec = c - p
        length = np.linalg.norm(dir_vec)
        if length < 1e-6:
            dir_vec = default_axis
            length = 0.0
        q = quaternion_from_vectors(default_axis, dir_vec)
        pitch, yaw, roll = quaternion_to_euler(q)
        scale = [length, 1.0, 1.0] if scale_axis == 'length' else [1.0, 1.0, 1.0]
        transforms[bone] = {
            'Location': loc,
            'Rotation': {'Pitch': pitch, 'Yaw': yaw, 'Roll': roll},
            'Scale': scale
        }
    return transforms

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
    
    # 본 매핑 정의
    bone_map = {
        'head': ('nose', 'neck'),
        'upperarm_l': ('left_shoulder', 'left_elbow'),
        'upperarm_r': ('right_shoulder', 'right_elbow'),
        'lowerarm_l': ('left_elbow', 'left_wrist'),
        'lowerarm_r': ('right_elbow', 'right_wrist'),
        'hand_l': ('left_wrist', 'left_pinky'),
        'hand_r': ('right_wrist', 'right_pinky'),
        'thigh_l': ('left_hip', 'left_knee'),
        'thigh_r': ('right_hip', 'right_knee'),
        'calf_l': ('left_knee', 'left_ankle'),
        'calf_r': ('right_knee', 'right_ankle'),
        'foot_l': ('left_ankle', 'left_heel'),
        'foot_r': ('right_ankle', 'right_heel'),
    }
    
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
                
                if not skeleton_sent:
                    send_skeleton_structure()
                    skeleton_sent = True
                
                # 3D 변환
                scale = 0.01  # 1cm 단위로 변환
                floor_angle = 0.0  # 바닥 각도
                origin_px = (width/2, height/2)  # 이미지 중심을 원점으로
                
                # 3D 좌표로 변환
                points_3d = map_2d_to_3d(points_2d, scale, floor_angle, origin_px)
                
                # 본 트랜스폼 계산
                bone_transforms = compute_bone_transforms_euler(
                    {f"point_{i}": point for i, point in enumerate(points_3d)},
                    bone_map,
                    scale_axis='length'
                )
                
                # 트랜스폼 리스트로 변환
                transform_list = []
                for bone in ['head', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 'lowerarm_r',
                           'hand_l', 'hand_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r',
                           'foot_l', 'foot_r']:
                    if bone in bone_transforms:
                        transform = bone_transforms[bone]
                        transform_list.append({
                            "Location": transform['Location'],
                            "Rotation": [transform['Rotation']['Pitch'], 
                                       transform['Rotation']['Yaw'], 
                                       transform['Rotation']['Roll'], 
                                       1.0],
                            "Scale": transform['Scale']
                        })
                
                send_frame_animation(transform_list)
                
                if user_data.use_frame:    
                    for point_2d in points_2d:
                        cv2.circle(frame, (int(point_2d[0]), int(point_2d[1])), 5, (0, 255, 0), -1)

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
            source_element += f"decodebin ! videoconvert ! "
            source_element += f"video/x-raw,format=RGB ! "
        
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