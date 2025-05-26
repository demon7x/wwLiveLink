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

# 스켈레톤 구조 메시지 전송 함수
def send_skeleton_structure():
    skeleton_message = {
        "mycharacter2": [
            {"Type": "CharacterSubject"},
            {"Name": "root", "Parent": "-1"},
            {"Name": "pelvis", "Parent": "0"},
            {"Name": "spine_01", "Parent": "1"},
            {"Name": "spine_02", "Parent": "2"},
            {"Name": "spine_03", "Parent": "3"},
            {"Name": "neck_01", "Parent": "4"},
            {"Name": "head", "Parent": "5"},
            {"Name": "clavicle_l", "Parent": "4"},
            {"Name": "upperarm_l", "Parent": "8"},
            {"Name": "lowerarm_l", "Parent": "9"},
            {"Name": "hand_l", "Parent": "10"},
            {"Name": "clavicle_r", "Parent": "4"},
            {"Name": "upperarm_r", "Parent": "12"},
            {"Name": "lowerarm_r", "Parent": "13"},
            {"Name": "hand_r", "Parent": "14"},
            {"Name": "thigh_l", "Parent": "1"},
            {"Name": "calf_l", "Parent": "16"},
            {"Name": "foot_l", "Parent": "17"},
            {"Name": "thigh_r", "Parent": "1"},
            {"Name": "calf_r", "Parent": "19"},
            {"Name": "foot_r", "Parent": "20"},
        ]
    }
    msg = json.dumps(skeleton_message).encode('utf-8')
    sock.sendto(msg, (UDP_IP, UDP_PORT))

# 프레임별 애니메이션 메시지 전송 함수
def send_frame_animation(bone_transforms):
    message = {
        "mycharacter2": [
            {"Type": "CharacterAnimation"},
            *bone_transforms
        ]
    }
    msg = json.dumps(message).encode('utf-8')
    sock.sendto(msg, (UDP_IP, UDP_PORT))
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
            # COCO → UE 본 매핑 (예시)
            # 실제 매핑 순서와 개수는 스켈레톤 구조와 맞춰야 함
            # points: [nose, left_eye, right_eye, ...] (COCO 순서)
            # 아래는 예시 매핑 (실제 본 구조에 맞게 수정 필요)
            bone_transforms = [
                {"Location": [0, 0, 0], "Rotation": [0, 0, 0, 1], "Scale": [1, 1, 1]},  # root
                {"Location": [points[11].x(), points[11].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]}, # pelvis (left_hip)
                {"Location": [points[5].x(), points[5].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]},  # spine_01 (left_shoulder)
                {"Location": [points[6].x(), points[6].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]},  # spine_02 (right_shoulder)
                {"Location": [points[0].x(), points[0].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]},  # spine_03 (nose)
                {"Location": [points[1].x(), points[1].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]},  # neck_01 (left_eye)
                {"Location": [points[2].x(), points[2].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]},  # head (right_eye)
                {"Location": [points[3].x(), points[3].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]},  # clavicle_l (left_ear)
                {"Location": [points[4].x(), points[4].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]},  # upperarm_l (right_ear)
                {"Location": [points[7].x(), points[7].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]},  # lowerarm_l (left_elbow)
                {"Location": [points[9].x(), points[9].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]},  # hand_l (left_wrist)
                {"Location": [points[8].x(), points[8].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]},  # clavicle_r (right_elbow)
                {"Location": [points[10].x(), points[10].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]}, # upperarm_r (right_wrist)
                {"Location": [points[12].x(), points[12].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]}, # lowerarm_r (right_hip)
                {"Location": [points[13].x(), points[13].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]}, # hand_r (left_knee)
                {"Location": [points[14].x(), points[14].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]}, # thigh_l (right_knee)
                {"Location": [points[15].x(), points[15].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]}, # calf_l (left_ankle)
                {"Location": [points[16].x(), points[16].y(), 0], "Rotation": [0,0,0,1], "Scale": [1,1,1]}, # foot_l (right_ankle)
                {"Location": [0,0,0], "Rotation": [0,0,0,1], "Scale": [1,1,1]}, # thigh_r (예시, 실제 매핑 필요)
                {"Location": [0,0,0], "Rotation": [0,0,0,1], "Scale": [1,1,1]}, # calf_r (예시)
                {"Location": [0,0,0], "Rotation": [0,0,0,1], "Scale": [1,1,1]}, # foot_r (예시)
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