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
import livelink_transform
import atexit

# UDP 설정 (필요시 수정)
UDP_IP = "192.168.0.255"  # 브로드캐스트 주소
UDP_PORT = 54321
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
skeleton_sent = False  # 스켈레톤 메시지 전송 여부 플래그
TRC_WRITER = None
TRC_RATE = 30.0

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


# ------------------------------------------------------------
# Dummy walk generator (Live Link-like bone transforms)
# ------------------------------------------------------------
def _zrot_quat(deg: float):
    rad = float(np.deg2rad(deg))
    s = np.sin(rad/2.0)
    c = np.cos(rad/2.0)
    return [0.0, 0.0, float(s), float(c)]

def _yrot_quat(deg: float):
    rad = float(np.deg2rad(deg))
    s = np.sin(rad/2.0)
    c = np.cos(rad/2.0)
    return [0.0, float(s), 0.0, float(c)]

def _xrot_quat(deg: float):
    rad = float(np.deg2rad(deg))
    s = np.sin(rad/2.0)
    c = np.cos(rad/2.0)
    return [float(s), 0.0, 0.0, float(c)]


def build_dummy_walk_transforms(t: float, swing_deg: float = 30.0) -> list:
    # Phases (left/right out of phase)
    phase = 2.0 * np.pi * t
    arm = swing_deg * np.sin(phase)
    arm_lo = 0.6 * swing_deg * np.sin(phase)
    wrist = 0.2 * swing_deg * np.sin(phase)
    # Legs: left/right opposite
    thigh_l = 30.0 * np.sin(phase)
    thigh_r = -30.0 * np.sin(phase)
    knee_l = 45.0 * max(0.0, -np.sin(phase))
    knee_r = 45.0 * max(0.0,  np.sin(phase))
    ankle = 8.0 * np.sin(phase)

    # Nominal bone lengths (units) so bones don't collapse at root
    BONE_LENGTH = {
        'head': 12.0,
        'upperarm_l': 28.0, 'upperarm_r': 28.0,
        'lowerarm_l': 26.0, 'lowerarm_r': 26.0,
        'hand_l': 10.0, 'hand_r': 10.0,
        'thigh_l': 40.0, 'thigh_r': 40.0,
        'calf_l': 43.0, 'calf_r': 43.0,
        'foot_l': 18.0, 'foot_r': 18.0,
    }

    transforms = []
    # Order must match skeleton: head, upperarm_l, upperarm_r, lowerarm_l, lowerarm_r,
    # hand_l, hand_r, thigh_l, thigh_r, calf_l, calf_r, foot_l, foot_r
    # Root(head)
    transforms.append({"Location":[0,0,0], "Rotation": _zrot_quat(0.0), "Scale":[1,1,1]})
    # Arms: use local +X offsets for all, rotate around Z for side swing
    transforms.append({"Location":[BONE_LENGTH['upperarm_l'],0,0], "Rotation": _zrot_quat(+arm), "Scale":[1,1,1]})          # upperarm_l
    transforms.append({"Location":[BONE_LENGTH['upperarm_r'],0,0], "Rotation": _zrot_quat(-arm), "Scale":[1,1,1]})          # upperarm_r
    transforms.append({"Location":[BONE_LENGTH['lowerarm_l'],0,0], "Rotation": _zrot_quat(+arm_lo), "Scale":[1,1,1]})       # lowerarm_l
    transforms.append({"Location":[BONE_LENGTH['lowerarm_r'],0,0], "Rotation": _zrot_quat(-arm_lo), "Scale":[1,1,1]})       # lowerarm_r
    transforms.append({"Location":[BONE_LENGTH['hand_l'],0,0], "Rotation": _zrot_quat(+wrist), "Scale":[1,1,1]})            # hand_l
    transforms.append({"Location":[BONE_LENGTH['hand_r'],0,0], "Rotation": _zrot_quat(-wrist), "Scale":[1,1,1]})            # hand_r
    # Legs: use local +X offsets, rotate around X for flexion/extension
    transforms.append({"Location":[BONE_LENGTH['thigh_l'],0,0], "Rotation": _xrot_quat(+thigh_l), "Scale":[1,1,1]})         # thigh_l
    transforms.append({"Location":[BONE_LENGTH['thigh_r'],0,0], "Rotation": _xrot_quat(+thigh_r), "Scale":[1,1,1]})         # thigh_r
    transforms.append({"Location":[BONE_LENGTH['calf_l'],0,0], "Rotation": _xrot_quat(+knee_l), "Scale":[1,1,1]})           # calf_l
    transforms.append({"Location":[BONE_LENGTH['calf_r'],0,0], "Rotation": _xrot_quat(+knee_r), "Scale":[1,1,1]})           # calf_r
    transforms.append({"Location":[BONE_LENGTH['foot_l'],0,0], "Rotation": _xrot_quat(+ankle), "Scale":[1,1,1]})            # foot_l
    transforms.append({"Location":[BONE_LENGTH['foot_r'],0,0], "Rotation": _xrot_quat(-ankle), "Scale":[1,1,1]})            # foot_r
    return transforms


def run_dummy_walk(fps: int = 30, speed_hz: float = 1.2, swing_deg: float = 30.0):
    global skeleton_sent
    if not skeleton_sent:
        send_skeleton_structure()
        skeleton_sent = True
    start = time.time()
    dt = 1.0 / max(1, int(fps))
    try:
        while True:
            now = time.time()
            t = (now - start) * float(speed_hz)  # cycles per second
            transforms = build_dummy_walk_transforms(t, swing_deg=swing_deg)
            send_frame_animation(transforms)
            time.sleep(dt)
    except KeyboardInterrupt:
        return

def map_2d_to_3d(
    keypoints_2d: np.ndarray,
    scale: float,
    floor_angle_deg: float,
    origin_px: tuple = (0, 0),
    depth: float = 0.0
) -> np.ndarray:
    """
    COCO 키포인트를 언리얼 엔진 좌표계로 변환
    - COCO: (0,0)이 좌상단, y축이 아래로
    - Unreal: (0,0,0)이 중심, y축이 앞으로, z축이 위로
    """
    # 이미지 좌표를 중심 기준으로 변환
    pts = keypoints_2d - np.array(origin_px)
    
    # 스케일 적용 및 좌표계 변환
    x_world = pts[:, 0] * 0.1424  # COCO x -> Unreal x (스케일 조정)
    y_world = np.zeros_like(x_world)  # COCO y -> Unreal y (전방)
    z_world = np.zeros_like(x_world)  # COCO z -> Unreal z (높이)
    
    return np.stack((x_world, y_world, z_world), axis=1)

def calculate_bone_rotation(parent_pos, child_pos, target_axis=[1,0,0], is_thigh=False):
    """
    부모-자식 본 사이의 회전을 쿼터니언으로 계산합니다.
    is_thigh: thigh 본인 경우 특별한 회전 처리
    """
    v = np.array(child_pos) - np.array(parent_pos)
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return [0,0,0,1]  # 기본 쿼터니언
    
    v_norm = v / norm
    
    if is_thigh:
        # thigh의 경우 언리얼 엔진의 좌표계에 맞게 회전 조정
        # 기본 방향을 [0, 0, -1]로 설정 (아래쪽 방향)
        target = np.array([0, 0, -1])
        
        # 벡터를 언리얼 엔진의 좌표계로 변환
        v_norm = np.array([v_norm[0], v_norm[1], -v_norm[2]])
        
        # -90도 회전 보정을 위한 추가 쿼터니언
        correction_angle = -np.pi / 2  # -90도
        correction_quat = [
            np.sin(correction_angle/2),  # x
            0,                           # y
            0,                           # z
            np.cos(correction_angle/2)   # w
        ]
    else:
        target = np.array(target_axis)
        correction_quat = [0, 0, 0, 1]  # 보정 없음
    
    # 두 벡터 사이의 회전축과 각도 계산
    axis = np.cross(v_norm, target)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-6:
        return [0,0,0,1]  # 벡터가 평행한 경우
    
    axis = axis / axis_norm
    dot = np.clip(np.dot(v_norm, target), -1.0, 1.0)
    angle = np.arccos(dot)
    
    # 쿼터니언 계산
    half_angle = angle / 2
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)
    
    # 기본 회전 쿼터니언
    base_quat = [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, cos_half]
    
    if is_thigh:
        # 쿼터니언 곱셈으로 90도 보정 적용
        q1 = base_quat
        q2 = correction_quat
        return [
            q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1],  # x
            q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0],  # y
            q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3],  # z
            q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2]   # w
        ]
    else:
        return base_quat

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
    스켈레톤 구조에 맞는 본 회전을 쿼터니언으로 계산합니다.
    """
    rotations = []
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
    
    # COCO 키포인트 인덱스 매핑
    keypoint_indices = {
        'head': 0,           # nose
        'upperarm_l': 5,     # left_shoulder
        'upperarm_r': 6,     # right_shoulder
        'lowerarm_l': 7,     # left_elbow
        'lowerarm_r': 8,     # right_elbow
        'hand_l': 9,         # left_wrist
        'hand_r': 10,        # right_wrist
        'thigh_l': 11,       # left_hip
        'thigh_r': 12,       # right_hip
        'calf_l': 13,        # left_knee
        'calf_r': 14,        # right_knee
        'foot_l': 15,        # left_ankle
        'foot_r': 16,        # right_ankle
    }
    
    # 각 본의 회전 계산
    for bone_name, parent_name in bone_hierarchy:
        if parent_name is None:
            # head는 기본 회전
            rotations.append([0,0,0,1])
        else:
            # 부모-자식 본 사이의 회전 계산
            parent_idx = keypoint_indices[parent_name]
            child_idx = keypoint_indices[bone_name]
            if parent_idx < len(points_3d) and child_idx < len(points_3d):
                # thigh 본인 경우 특별한 회전 처리
                is_thigh = bone_name in ['thigh_l', 'thigh_r']
                rot = calculate_bone_rotation(points_3d[parent_idx], points_3d[child_idx], is_thigh=is_thigh)
                rotations.append(rot)
            else:
                rotations.append([0,0,0,1])
    
    return rotations

def get_bone_positions(points_3d):
    """
    스켈레톤 구조에 맞는 본 위치를 계산합니다.
    """
    positions = []
    # COCO 키포인트 인덱스 매핑
    keypoint_indices = {
        'head': 0,           # nose
        'upperarm_l': 5,     # left_shoulder
        'upperarm_r': 6,     # right_shoulder
        'lowerarm_l': 7,     # left_elbow
        'lowerarm_r': 8,     # right_elbow
        'hand_l': 9,         # left_wrist
        'hand_r': 10,        # right_wrist
        'thigh_l': 11,       # left_hip
        'thigh_r': 12,       # right_hip
        'calf_l': 13,        # left_knee
        'calf_r': 14,        # right_knee
        'foot_l': 15,        # left_ankle
        'foot_r': 16,        # right_ankle
    }
    
    # 각 본의 위치 추출
    for bone_name in ['head', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 'lowerarm_r',
                     'hand_l', 'hand_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r',
                     'foot_l', 'foot_r']:
        idx = keypoint_indices[bone_name]
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
                
                if not skeleton_sent:
                    send_skeleton_structure()
                    skeleton_sent = True
                
                # 외부 모듈로 3D 보정 및 Live Link 변환 수행
                bone_transforms = livelink_transform.compute_transforms(
                    keypoints_2d=points_2d,
                    width=width,
                    height=height,
                    height_m=getattr(args, 's2d_height', None),
                    floor_angle_deg=getattr(args, 's2d_floor', None),
                    direction=getattr(args, 's2d_direction', 'side'),
                )
                
                send_frame_animation(bone_transforms)

                # TRC 저장: 변환된 3D 포인트(COCO 17)를 기록
                if TRC_WRITER is not None:
                    pts3d = livelink_transform.compute_points3d(
                        keypoints_2d=points_2d,
                        width=width,
                        height=height,
                        height_m=getattr(args, 's2d_height', None),
                        floor_angle_deg=getattr(args, 's2d_floor', None),
                        direction=getattr(args, 's2d_direction', 'side'),
                    )
                    TRC_WRITER.write_frame(pts3d)
                
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
    # Dummy walk options
    parser.add_argument('--dummy-walk', action='store_true', help='Send synthetic walk Live Link data instead of running pipeline')
    parser.add_argument('--dummy-fps', type=int, default=30, help='Dummy walk FPS')
    parser.add_argument('--dummy-speed', type=float, default=1.2, help='Dummy walk speed in cycles/sec')
    parser.add_argument('--dummy-swing', type=float, default=30.0, help='Arm swing amplitude in degrees')
    # Sports2D-style options
    parser.add_argument('--s2d-height', type=float, default=None, help='Subject height in meters for scale (Sports2D-like)')
    parser.add_argument('--s2d-floor', type=float, default=None, help='Floor angle in degrees (override auto)')
    parser.add_argument('--s2d-direction', type=str, default='side', choices=['side','front','back'], help='Camera view relative to subject')
    # TRC export options
    parser.add_argument('--save-trc', nargs='?', const='@output.trc', default=None, help='Path to save streaming TRC of mapped 3D COCO points (default @output.trc if no path)')
    parser.add_argument('--trc-rate', type=float, default=30.0, help='TRC DataRate (Hz)')
    args = parser.parse_args()
    # init TRC writer if requested
    if getattr(args, 'save_trc', None):
        TRC_RATE = float(getattr(args, 'trc_rate', 30.0))
        TRC_WRITER = livelink_transform.TRCWriter(
            filepath=args.save_trc,
            marker_names=livelink_transform.COCO_NAMES,
            data_rate=TRC_RATE,
            units='mm'
        )
        # finalize header on exit
        atexit.register(lambda: TRC_WRITER.finalize() if TRC_WRITER is not None else None)
    if getattr(args, 'dummy_walk', False):
        run_dummy_walk(fps=args.dummy_fps, speed_hz=args.dummy_speed, swing_deg=args.dummy_swing)
    else:
        app = GStreamerPoseEstimationApp(args, user_data)
        app.run()