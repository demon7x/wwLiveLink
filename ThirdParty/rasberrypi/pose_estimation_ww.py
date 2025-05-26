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

# T-Pose 기준 본별 트랜스폼 (X, Y, Z)
TPOSE_BONE_LOCATIONS = [
    [-0.000001, 0.138235, 162.503119],   # head
    [16.055346, -2.507836, 142.836729],  # upperarm_l
    [-16.055299, -2.507837, 142.836686], # upperarm_r
    [43.121829, -3.260803, 143.688663],  # lowerarm_l
    [-43.121353, -3.260791, 143.688607], # lowerarm_r
    [69.100656, -0.937592, 144.501407],  # hand_l
    [-69.100513, -0.937534, 144.501369], # hand_r
    [11.154586, 2.650447, 95.471892],    # thigh_l
    [-11.154600, 2.650411, 95.471838],   # thigh_r
    [13.062653, 1.697236, 49.769600],    # calf_l
    [-13.062663, 1.697201, 49.769644],   # calf_r
    [14.684443, 0.041375, 8.128633],     # foot_l
    [-14.684455, 0.041339, 8.128632],    # foot_r
]

# Parent 인덱스 (skeleton_message 순서와 일치)
PARENT_INDICES = [
    -1,  # head
    0,   # upperarm_l
    0,   # upperarm_r
    1,   # lowerarm_l
    2,   # lowerarm_r
    3,   # hand_l
    4,   # hand_r
    0,   # thigh_l
    0,   # thigh_r
    7,   # calf_l
    8,   # calf_r
    9,   # foot_l
    10,  # foot_r
]

# T-Pose Local 위치값 (예시)
TPOSE_LOCAL_LOCATIONS = [
    [5.758485, 0.000000, 0.000000],      # head
    [15.286094, 0.000000, 0.000000],     # upperarm_l
    [-15.285989, 0.000005, -0.000402],   # upperarm_r
    [27.090353, -0.000000, -0.000000],   # lowerarm_l
    [-27.089924, 0.000000, -0.000000],   # lowerarm_r
    [26.095160, -0.000000, 0.000000],    # hand_l
    [-26.095495, 0.000000, 0.000000],    # hand_r
    [-3.231992, 0.068032, -11.154586],   # thigh_l
    [-3.232044, 0.067992, 11.154600],    # thigh_r
    [-45.752037, -0.000000, 0.000000],   # calf_l
    [45.751938, 0.000000, -0.000000],    # calf_r
    [-41.705421, -0.000000, -0.000000],  # foot_l
    [41.705467, 0.000000, 0.000000],     # foot_r
]

def world_to_local_positions(bone_world_positions, parent_indices):
    bone_local_positions = []
    for i, pos in enumerate(bone_world_positions):
        parent_idx = parent_indices[i]
        if parent_idx == -1:
            bone_local_positions.append(pos)
        else:
            parent_pos = bone_world_positions[parent_idx]
            local_pos = [p - q for p, q in zip(pos, parent_pos)]
            bone_local_positions.append(local_pos)
    return bone_local_positions

def world_to_local_with_tpose(bone_world_positions, tpose_world_positions, tpose_local_positions, parent_indices):
    bone_local_positions = []
    for i, pos in enumerate(bone_world_positions):
        parent_idx = parent_indices[i]
        if parent_idx == -1:
            bone_local_positions.append(tpose_local_positions[i])
        else:
            parent_pos = bone_world_positions[parent_idx]
            tpose_parent_pos = tpose_world_positions[parent_idx]
            delta = [ (p - q) - (tp - tq) for p, q, tp, tq in zip(pos, parent_pos, tpose_world_positions[i], tpose_parent_pos) ]
            local_pos = [t + d for t, d in zip(tpose_local_positions[i], delta)]
            bone_local_positions.append(local_pos)
    return bone_local_positions

def yolo_to_unreal_relative(x, y, img_w, img_h, tpose_loc, scale=0.1):
    x_pixel = x * img_w
    y_pixel = y * img_h
    x_unreal = (x_pixel - img_w / 2) * scale + tpose_loc[0]
    y_unreal = - (y_pixel - img_h / 2) * scale + tpose_loc[1]
    z_unreal = tpose_loc[2]
    return [x_unreal, y_unreal, z_unreal]

def normalize_and_scale(points, key_height_cm=160):
    # points: [[x, y, 0], ...] (정규화 0~1)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_center = sum(xs) / len(xs)
    y_top = min(ys)
    y_bottom = max(ys)
    y_height = y_bottom - y_top if y_bottom > y_top else 1e-5
    scale = key_height_cm / y_height / 10.0
    result = []
    for x, y, _ in points:
        x_new = (x - x_center) * scale
        z_new = (1 - y) * scale  # y축 반전 후 z로
        y_new = 0  # 2D이므로 y는 0(혹은 필요시 좌우로 이동)
        result.append([x_new, y_new, z_new])
    return result

def two_bone_ik(start, mid, end):
    upper = np.array(mid) - np.array(start)
    lower = np.array(end) - np.array(mid)
    if np.linalg.norm(upper) < 1e-6 or np.linalg.norm(lower) < 1e-6:
        return [0,0,0,1], [0,0,0,1]
    upper_norm = upper / np.linalg.norm(upper)
    lower_norm = lower / np.linalg.norm(lower)
    rot_shoulder = R.align_vectors([upper_norm], [[1,0,0]])[0].as_quat().tolist()
    rot_elbow = R.align_vectors([lower_norm], [upper_norm])[0].as_quat().tolist()
    return rot_shoulder, rot_elbow

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    global skeleton_sent
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK
        
    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"
    
    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Parse the detections
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            string_to_print += (f"Detection: {label} {confidence:.2f}\n")
            # Pose estimation landmarks from detection (if available)
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                left_eye = points[1]  # assuming 1 is the index for the left eye
                right_eye = points[2]  # assuming 2 is the index for the right eye
                # The landmarks are normalized to the bounding box, we also need to convert them to the frame size
                left_eye_x = int((left_eye.x() * bbox.width() + bbox.xmin()) * width)
                left_eye_y = int((left_eye.y() * bbox.height() + bbox.ymin()) * height)
                right_eye_x = int((right_eye.x() * bbox.width() + bbox.xmin()) * width)
                right_eye_y = int((right_eye.y() * bbox.height() + bbox.ymin()) * height)
                string_to_print += (f" Left eye: x: {left_eye_x:.2f} y: {left_eye_y:.2f} Right eye: x: {right_eye_x:.2f} y: {right_eye_y:.2f}\n")
                string_to_print += (f" origin_data: x: {left_eye} \n")
                if user_data.use_frame:    
                    # Add markers to the frame to show eye landmarks
                    cv2.circle(frame, (left_eye_x, left_eye_y), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (right_eye_x, right_eye_y), 5, (0, 255, 0), -1)
                    # Note: using imshow will not work here, as the callback function is not running in the main thread   
            
            # 스켈레톤 메시지 최초 1회 전송
            if not skeleton_sent:
                send_skeleton_structure()
                skeleton_sent = True
            # points에서 x, y만 추출하여 변환 함수 적용
            raw_points = [
                [points[0].x(), points[0].y(), 0],   # head (nose)
                [points[5].x(), points[5].y(), 0],   # upperarm_l (left_shoulder)
                [points[6].x(), points[6].y(), 0],   # upperarm_r (right_shoulder)
                [points[7].x(), points[7].y(), 0],   # lowerarm_l (left_elbow)
                [points[8].x(), points[8].y(), 0],   # lowerarm_r (right_elbow)
                [points[9].x(), points[9].y(), 0],   # hand_l (left_wrist)
                [points[10].x(), points[10].y(), 0], # hand_r (right_wrist)
                [points[11].x(), points[11].y(), 0], # thigh_l (left_hip)
                [points[12].x(), points[12].y(), 0], # thigh_r (right_hip)
                [points[13].x(), points[13].y(), 0], # calf_l (left_knee)
                [points[14].x(), points[14].y(), 0], # calf_r (right_knee)
                [points[15].x(), points[15].y(), 0], # foot_l (left_ankle)
                [points[16].x(), points[16].y(), 0], # foot_r (right_ankle)
            ]
            bone_world_positions = normalize_and_scale(raw_points, key_height_cm=160)
            # IK 적용: 왼팔(upperarm_l, lowerarm_l, hand_l)
            rot_upperarm_l, rot_lowerarm_l = two_bone_ik(
                bone_world_positions[1], bone_world_positions[3], bone_world_positions[5]
            )
            # IK 적용: 오른팔(upperarm_r, lowerarm_r, hand_r)
            rot_upperarm_r, rot_lowerarm_r = two_bone_ik(
                bone_world_positions[2], bone_world_positions[4], bone_world_positions[6]
            )
            # 나머지 본은 회전 [0,0,0,1]
            bone_rotations = [
                [0,0,0,1],         # head
                rot_upperarm_l,    # upperarm_l
                rot_upperarm_r,    # upperarm_r
                rot_lowerarm_l,    # lowerarm_l
                rot_lowerarm_r,    # lowerarm_r
                [0,0,0,1],        # hand_l
                [0,0,0,1],        # hand_r
                [0,0,0,1],        # thigh_l
                [0,0,0,1],        # thigh_r
                [0,0,0,1],        # calf_l
                [0,0,0,1],        # calf_r
                [0,0,0,1],        # foot_l
                [0,0,0,1],        # foot_r
            ]
            bone_local_positions = world_to_local_positions(bone_world_positions, PARENT_INDICES)
            bone_transforms = [
                {"Location": bone_local_positions[i], "Rotation": bone_rotations[i], "Scale": [1,1,1]}
                for i in range(len(bone_local_positions))
            ]
            send_frame_animation(bone_transforms)

    if user_data.use_frame:
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK
    


# This function can be used to get the COCO keypoints coorespondence map
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    keypoints = {
        'nose': 1,
        'left_eye': 2,
        'right_eye': 3,
        'left_ear': 4,
        'right_ear': 5,
        'left_shoulder': 6,
        'right_shoulder': 7,
        'left_elbow': 8,
        'right_elbow': 9,
        'left_wrist': 10,
        'right_wrist': 11,
        'left_hip': 12,
        'right_hip': 13,
        'left_knee': 14,
        'right_knee': 15,
        'left_ankle': 16,
        'right_ankle': 17,
    }

    return keypoints
#-----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class

class GStreamerPoseEstimationApp(GStreamerApp):
    def __init__(self, args, user_data):
        # Call the parent class constructor
        super().__init__(args, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        self.default_postprocess_so = os.path.join(self.postprocess_dir, 'libyolov8pose_post.so')
        self.post_function_name = "filter"
        self.hef_path = os.path.join(self.current_path, 'yolov8s_pose_h8l_pi.hef')
        self.app_callback = app_callback
        
        # Set the process title
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
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    parser = get_default_parser()
    args = parser.parse_args()
    app = GStreamerPoseEstimationApp(args, user_data)
    app.run()