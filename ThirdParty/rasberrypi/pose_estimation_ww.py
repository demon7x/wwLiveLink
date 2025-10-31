import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
 
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

def _mul_quat(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def build_dummy_walk_transforms(t: float, swing_deg: float = 30.0) -> list:
    # 일반 걷기 사이클
    phase = 2.0 * np.pi * t
    arm = swing_deg * np.sin(phase)
    arm_lo = 0.6 * swing_deg * np.sin(phase + np.pi/8.0)
    leg = 35.0 * np.sin(phase)
    knee_l = 40.0 * max(0.0, -np.sin(phase))
    knee_r = 40.0 * max(0.0,  np.sin(phase))
    ankle = 8.0 * np.sin(phase + np.pi/4.0)

    # nominal lengths
    BONE_LENGTH = {
        'upperarm_l': 28.0, 'upperarm_r': 28.0,
        'lowerarm_l': 26.0, 'lowerarm_r': 26.0,
        'hand_l': 10.0, 'hand_r': 10.0,
        'thigh_l': 40.0, 'thigh_r': 40.0,
        'calf_l': 43.0, 'calf_r': 43.0,
        'foot_l': 18.0, 'foot_r': 18.0,
    }

    transforms = []
    # order: head, upperarm_l, upperarm_r, lowerarm_l, lowerarm_r,
    # hand_l, hand_r, thigh_l, thigh_r, calf_l, calf_r, foot_l, foot_r
    transforms.append({"Location":[0,0,0], "Rotation": _zrot_quat(0.0), "Scale":[1,1,1]})                             # head
    # arms (Y swing)
    transforms.append({"Location":[BONE_LENGTH['upperarm_l'],0,0], "Rotation": _yrot_quat(+arm), "Scale":[1,1,1]})   # upperarm_l
    transforms.append({"Location":[BONE_LENGTH['upperarm_r'],0,0], "Rotation": _yrot_quat(-arm), "Scale":[1,1,1]})   # upperarm_r
    transforms.append({"Location":[BONE_LENGTH['lowerarm_l'],0,0], "Rotation": _yrot_quat(+arm_lo), "Scale":[1,1,1]})# lowerarm_l
    transforms.append({"Location":[BONE_LENGTH['lowerarm_r'],0,0], "Rotation": _yrot_quat(-arm_lo), "Scale":[1,1,1]})# lowerarm_r
    transforms.append({"Location":[BONE_LENGTH['hand_l'],0,0], "Rotation": _yrot_quat(0.2*arm_lo), "Scale":[1,1,1]}) # hand_l
    transforms.append({"Location":[BONE_LENGTH['hand_r'],0,0], "Rotation": _yrot_quat(-0.2*arm_lo), "Scale":[1,1,1]})# hand_r
    # legs
    transforms.append({"Location":[BONE_LENGTH['thigh_l'],0,0], "Rotation": _yrot_quat(-leg), "Scale":[1,1,1]})      # thigh_l
    transforms.append({"Location":[BONE_LENGTH['thigh_r'],0,0], "Rotation": _yrot_quat(+leg), "Scale":[1,1,1]})      # thigh_r
    transforms.append({"Location":[BONE_LENGTH['calf_l'],0,0], "Rotation": _xrot_quat(+knee_l), "Scale":[1,1,1]})    # calf_l
    transforms.append({"Location":[BONE_LENGTH['calf_r'],0,0], "Rotation": _xrot_quat(+knee_r), "Scale":[1,1,1]})    # calf_r
    transforms.append({"Location":[BONE_LENGTH['foot_l'],0,0], "Rotation": _xrot_quat(+ankle), "Scale":[1,1,1]})     # foot_l
    transforms.append({"Location":[BONE_LENGTH['foot_r'],0,0], "Rotation": _xrot_quat(-ankle), "Scale":[1,1,1]})     # foot_r
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