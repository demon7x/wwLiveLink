import numpy as np


# COCO keypoint indices used in this project (17-point layout)
KEYPOINT_INDICES = {
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

# COCO name→index (convenience for scale estimation etc.)
COCO = {
    "nose":0, "left_eye":1, "right_eye":2, "left_ear":3, "right_ear":4,
    "left_shoulder":5, "right_shoulder":6, "left_elbow":7, "right_elbow":8,
    "left_wrist":9, "right_wrist":10, "left_hip":11, "right_hip":12,
    "left_knee":13, "right_knee":14, "left_ankle":15, "right_ankle":16
}

# Ordered COCO names for TRC export (17 markers)
COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Bone order expected by the sending side/skeleton definition
BONE_ORDER = [
    'head',
    'upperarm_l', 'upperarm_r',
    'lowerarm_l', 'lowerarm_r',
    'hand_l', 'hand_r',
    'thigh_l', 'thigh_r',
    'calf_l', 'calf_r',
    'foot_l', 'foot_r',
]


def normalize_keypoints_2d_yolov8(points_2d: np.ndarray, frame_width: int, frame_height: int, person_height_m: float = 1.65) -> np.ndarray:
    """
    YOLOv8 (17,2) 키포인트를 Sports2D 스타일 3D로 정규화 (단일 프레임)
    - 입력: points_2d (17,2) 혹은 (17,>=2)
    - 출력: (17,3) [X,Y,Z] (미터 단위, Z=0)
    """
    if points_2d.shape[0] < 13:
        # 관절 수 부족 시 영벡터 반환
        return np.zeros((17, 3), dtype=float)

    # 좌/우 엉덩이 거리 기반 스케일(Colab 예시 로직을 그대로 따름)
    l_hip = points_2d[11][:2]
    r_hip = points_2d[12][:2]
    hip_width_px = float(np.linalg.norm(l_hip - r_hip))
    if hip_width_px < 10.0:
        # 너무 짧으면 실패로 간주하고 영벡터 반환
        return np.zeros((17, 3), dtype=float)

    scale = float(person_height_m) / (hip_width_px * 2.5)

    out = np.zeros((17, 3), dtype=float)
    cx = frame_width / 2.0
    cy = frame_height / 2.0
    for i in range(min(17, points_2d.shape[0])):
        x_px = float(points_2d[i, 0])
        y_px = float(points_2d[i, 1])
        x_norm = (x_px - cx) / frame_width
        y_norm = (y_px - cy) / frame_height
        x_m = x_norm * frame_width * scale
        y_m = -y_norm * frame_height * scale  # 위로 갈수록 +Y
        out[i] = [x_m, y_m, 0.0]
    return out


def rotate_to_ground_plane(points_3d: np.ndarray, angle_deg: float = 90.0) -> np.ndarray:
    """
    X축 기준으로 angle_deg만큼 회전시켜 지면 정렬 (Colab 예시와 동일)
    - 입력: (17,3)
    - 출력: (17,3)
    """
    angle_rad = np.deg2rad(float(angle_deg))
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, np.cos(angle_rad), -np.sin(angle_rad)],
                  [0.0, np.sin(angle_rad),  np.cos(angle_rad)]], dtype=float)
    return (points_3d @ R.T)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


# 위 Colab 기반 파이프라인만 사용하므로 추가 스케일/카메라 추정 함수는 제거


# 바닥 각도 추정 등 추가 보정은 현재 사용하지 않음


# 기존 map_2d_to_3d는 사용하지 않음 (Colab 스타일 고정 파이프라인 사용)


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> list:
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return [0.0, 0.0, 0.0, 1.0]
    axis = axis / axis_norm
    half = angle * 0.5
    s = np.sin(half)
    return [axis[0] * s, axis[1] * s, axis[2] * s, float(np.cos(half))]


def _mul_quat(q1: list, q2: list) -> list:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def calculate_bone_rotation(parent_pos: np.ndarray, child_pos: np.ndarray, *, is_thigh: bool = False) -> list:
    """
    Calculate quaternion [x, y, z, w] aligning default bone axis to the vector child-parent.
    Default bone axis: +X. For thighs we apply a corrective rotation to match Unreal rig expectations.
    """
    vec = np.array(child_pos) - np.array(parent_pos)
    n = np.linalg.norm(vec)
    if n < 1e-6:
        return [0.0, 0.0, 0.0, 1.0]
    v = vec / n

    # Align +X to v
    x_axis = np.array([1.0, 0.0, 0.0])
    axis = np.cross(x_axis, v)
    dot = np.clip(np.dot(x_axis, v), -1.0, 1.0)
    angle = float(np.arccos(dot))
    base_q = _quat_from_axis_angle(axis, angle)

    if is_thigh:
        # Apply a corrective rotation around X to account for rig orientation (tune if needed)
        # Start with -90 degrees, user reported further flips; expose as -90 by default
        correction = _quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -np.pi / 2.0)
        return _mul_quat(base_q, correction)
    return base_q


# 위치 리스트 반환 함수는 사용하지 않음


# 회전 리스트 단독 반환 함수는 사용하지 않음 (compute_transforms에서 직접 계산)


def compute_points3d(keypoints_2d: np.ndarray, width: int, height: int, *, K: tuple | None = None, height_m: float | None = None, floor_angle_deg: float | None = None, direction: str = 'side') -> np.ndarray:
    """
    Colab(sport2d) 스타일 변환:
      1) normalize_keypoints_2d_yolov8(points_2d, width, height)
      2) rotate_to_ground_plane(..., angle=90deg)
    반환: (17,3)
    """
    pts3d = normalize_keypoints_2d_yolov8(keypoints_2d, width, height, person_height_m=1.65)
    pts3d = rotate_to_ground_plane(pts3d, angle_deg=90.0)
    return pts3d


class TRCWriter:
    """
    Minimal TRC writer for streaming. Writes header once, then appends frames.
    NumFrames in header is set to 0 (unknown). Tools usually tolerate it for prototyping.
    """
    def __init__(self, filepath: str, marker_names: list[str], data_rate: float = 30.0, units: str = 'mm'):
        self.filepath = filepath
        self.marker_names = marker_names
        self.data_rate = float(data_rate)
        self.units = units
        self.frame_index = 0
        # Match reference output scaling: write raw meters even if header says 'mm'
        # (User's reference file uses 'mm' in header but values are ~meters.)
        self._scale_out = 1.0
        self._ensure_header()

    def _ensure_header(self):
        if self.frame_index > 0:
            return
        with open(self.filepath, 'w', encoding='utf-8') as f:
            fname = self.filepath.split('/')[-1]
            f.write(f"PathFileType\t4\t(X/Y/Z)\t{fname}\n")
            f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
            # Use integer formatting to mirror the reference TRC header style
            rate_i = int(round(self.data_rate))
            f.write(f"{rate_i}\t{rate_i}\t0\t{len(self.marker_names)}\t{self.units}\t{rate_i}\t1\t0\n")
            # Columns
            f.write("Frame#\tTime\t" + "\t".join([f"{n}\t\t" for n in self.marker_names]).rstrip('\t') + "\n")
            # Sub-columns X Y Z for each marker
            xyz_labels = []
            for i in range(1, len(self.marker_names)+1):
                xyz_labels.append(f"X{i}\tY{i}\tZ{i}")
            f.write("\t\t" + "\t\t\t".join(xyz_labels) + "\n")

    def write_frame(self, points_3d: np.ndarray):
        """
        points_3d: (N,3) in marker order matching marker_names.
        Time computed from data_rate.
        """
        N = len(self.marker_names)
        if points_3d.shape[0] < N:
            # pad missing markers with zeros
            pad = np.zeros((N - points_3d.shape[0], 3), dtype=float)
            pts = np.vstack([points_3d, pad])
        else:
            pts = points_3d[:N]
        t = self.frame_index / max(self.data_rate, 1e-6)
        cols = []
        for i in range(N):
            x, y, z = float(pts[i, 0])*self._scale_out, float(pts[i, 1])*self._scale_out, float(pts[i, 2])*self._scale_out
            cols.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"{self.frame_index+1:d}\t{t:.6f}\t" + "\t".join(cols) + "\n")
        self.frame_index += 1

    def finalize(self):
        """Rewrite NumFrames and OrigNumFrames after streaming ends."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) < 3:
                return
            # second line: DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames
            parts = lines[1].strip().split('\t')
            if len(parts) >= 8:
                parts[2] = str(self.frame_index)  # NumFrames
                parts[7] = str(self.frame_index)  # OrigNumFrames
                lines[1] = "\t".join(parts) + "\n"
                with open(self.filepath, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
        except Exception:
            pass

def compute_transforms(keypoints_2d: np.ndarray, width: int, height: int, *, K: tuple | None = None, height_m: float | None = None, floor_angle_deg: float | None = None, direction: str = 'side') -> list:
    """
    Colab(sport2d) 스타일 3D 포인트로부터 Live Link 로컬 본 트랜스폼 생성:
      - 루트(head) Location=[0,0,0], Rotation=identity
      - 자식 본 Location=[length,0,0], Rotation=+X를 (child-parent) 방향으로 정렬
    """
    points_3d = compute_points3d(keypoints_2d, width, height)

    parents = {
        'head': None,
        'upperarm_l': 'head',
        'upperarm_r': 'head',
        'lowerarm_l': 'upperarm_l',
        'lowerarm_r': 'upperarm_r',
        'hand_l': 'lowerarm_l',
        'hand_r': 'lowerarm_r',
        'thigh_l': 'head',
        'thigh_r': 'head',
        'calf_l': 'thigh_l',
        'calf_r': 'thigh_r',
        'foot_l': 'calf_l',
        'foot_r': 'calf_r',
    }

    transforms = []
    for bone in BONE_ORDER:
        parent = parents[bone]
        if parent is None:
            loc = [0.0, 0.0, 0.0]
            rot = [0.0, 0.0, 0.0, 1.0]
        else:
            c_idx = KEYPOINT_INDICES[bone]
            p_idx = KEYPOINT_INDICES[parent]
            if c_idx < len(points_3d) and p_idx < len(points_3d):
                vec = points_3d[c_idx] - points_3d[p_idx]
                length = float(np.linalg.norm(vec))
                loc = [length, 0.0, 0.0]
                rot = calculate_bone_rotation(points_3d[p_idx], points_3d[c_idx], is_thigh=(bone in ['thigh_l','thigh_r']))
            else:
                loc = [0.0, 0.0, 0.0]
                rot = [0.0, 0.0, 0.0, 1.0]

        transforms.append({
            "Location": loc,
            "Rotation": rot,
            "Scale": [1.0, 1.0, 1.0],
        })

    return transforms


