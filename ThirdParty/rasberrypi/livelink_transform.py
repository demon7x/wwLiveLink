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


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def _backproject_ray(u: float, v: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = (u - cx) / fx
    y = (v - cy) / fy
    d = np.array([x, y, 1.0], dtype=np.float32)
    return _unit(d)


def _transform_to_ue(P: np.ndarray) -> np.ndarray:
    """
    Camera(OpenCV: +X right, +Y down, +Z forward) → UE(+X forward, +Y right, +Z up)
    """
    M = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0,-1, 0]], dtype=np.float32)
    return (M @ P.T).T


def _estimate_scale_m_per_px(kps2d: np.ndarray, *, height_m: float | None = None) -> float:
    """
    Rough m/px from shoulder/hip/neck-pelvis heuristics. If height_m provided,
    scale so that head→ankle_mid pixel distance matches height_m.
    """
    def pix_dist(i: int, j: int) -> float:
        ui,vi = kps2d[i]
        uj,vj = kps2d[j]
        return float(np.hypot(ui-uj, vi-vj) + 1e-6)

    l_sh, r_sh = COCO["left_shoulder"], COCO["right_shoulder"]
    l_hip, r_hip = COCO["left_hip"], COCO["right_hip"]

    neck_uv = 0.5 * (kps2d[l_sh,:2] + kps2d[r_sh,:2])
    pelvis_uv = 0.5 * (kps2d[l_hip,:2] + kps2d[r_hip,:2])
    neck_pelvis_px = float(np.hypot(*(neck_uv - pelvis_uv)))

    pairs = []
    # Typical adult metrics (meters) — tune for your dataset
    spine_len_m = 0.45  # pelvis->neck
    shoulder_width_m = 0.36
    hip_width_m = 0.30
    if neck_pelvis_px > 2.0:
        pairs.append((neck_pelvis_px, spine_len_m))
    shoulder_px = pix_dist(l_sh, r_sh)
    if shoulder_px > 2.0:
        pairs.append((shoulder_px, shoulder_width_m))
    hip_px = pix_dist(l_hip, r_hip)
    if hip_px > 2.0:
        pairs.append((hip_px, hip_width_m))

    # Height-based scale if requested
    if height_m is not None and height_m > 0:
        nose = COCO["nose"]; l_ank = COCO["left_ankle"]; r_ank = COCO["right_ankle"]
        ankle_mid = 0.5 * (kps2d[l_ank,:2] + kps2d[r_ank,:2])
        h_px = float(np.hypot(*(kps2d[nose,:2] - ankle_mid)))
        if h_px > 2.0:
            return float(height_m) / h_px

    ratios = [px / m for (px, m) in pairs if m > 1e-6]
    if not ratios:
        return 0.002  # 1px≈2mm fallback
    px_per_m = float(np.median(ratios))
    return 1.0 / max(px_per_m, 1e-6)


def _detect_floor_angle(points_2d: np.ndarray) -> float:
    """Estimate floor angle (radians) from ankle line orientation."""
    l_ank = COCO["left_ankle"]; r_ank = COCO["right_ankle"]
    v = points_2d[r_ank,:2] - points_2d[l_ank,:2]
    return float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 1e-6 else 0.0


def map_2d_to_3d(points_2d: np.ndarray, width: int, height: int, *, K: tuple | None = None, init_z_m: float = 2.5, height_m: float | None = None, floor_angle_deg: float | None = None, direction: str = 'side') -> np.ndarray:
    """
    Two modes:
      - If K (fx,fy,cx,cy) provided: lift pixel rays to 3D at nominal depth (camera model), transform to UE.
      - Else: Sports2D-style planar mapping: rotate by floor angle, convert px→meters, map to UE plane (Y=0).
    Returns (N,3) UE-space points.
    """
    if K is not None:
        fx, fy, cx, cy = K
        rays = np.stack([
            _backproject_ray(float(u), float(v), fx, fy, cx, cy) for (u, v) in points_2d
        ], axis=0)
        pts3d_cam = rays * float(init_z_m)
        pts3d_ue = _transform_to_ue(pts3d_cam)
        m_per_px = _estimate_scale_m_per_px(points_2d, height_m=height_m)
        pts3d_ue[:, 0] *= float(m_per_px)
        pts3d_ue[:, 1] *= float(m_per_px)
        return pts3d_ue

    # Sports2D-style planar mapping
    origin_px = np.array([width/2.0, height/2.0], dtype=np.float32)
    pts = points_2d[:, :2] - origin_px
    floor_angle = np.deg2rad(float(floor_angle_deg)) if floor_angle_deg is not None else _detect_floor_angle(points_2d)
    c, s = np.cos(-floor_angle), np.sin(-floor_angle)  # rotate to align floor horizontally
    R2 = np.array([[c,-s],[s,c]], dtype=np.float32)
    pr = (R2 @ pts.T).T  # rotated pixels
    m_per_px = _estimate_scale_m_per_px(points_2d, height_m=height_m)
    x = pr[:, 0] * float(m_per_px)
    z = -pr[:, 1] * float(m_per_px)  # pixel down -> Z up
    # Directional normative depth (very small offset) to help UE orientation if needed
    if direction in ('front', 'back'):
        sign = 1.0 if direction == 'front' else -1.0
        y = np.full_like(x, 0.0) + 0.02 * sign
    else:
        y = np.zeros_like(x)
    return np.stack((x, y, z), axis=1)


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


def get_bone_positions(points_3d: np.ndarray) -> list:
    positions = []
    for bone in BONE_ORDER:
        idx = KEYPOINT_INDICES[bone]
        if idx < len(points_3d):
            positions.append(points_3d[idx].tolist())
        else:
            positions.append([0.0, 0.0, 0.0])
    return positions


def get_bone_rotations(points_3d: np.ndarray) -> list:
    # Define simple hierarchy mapping for parent of each bone by name
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

    rotations = []
    for bone in BONE_ORDER:
        parent = parents[bone]
        if parent is None:
            rotations.append([0.0, 0.0, 0.0, 1.0])
            continue
        c_idx = KEYPOINT_INDICES[bone]
        p_idx = KEYPOINT_INDICES[parent]
        if c_idx < len(points_3d) and p_idx < len(points_3d):
            rotations.append(
                calculate_bone_rotation(points_3d[p_idx], points_3d[c_idx], is_thigh=(bone in ['thigh_l', 'thigh_r']))
            )
        else:
            rotations.append([0.0, 0.0, 0.0, 1.0])
    return rotations


def compute_points3d(keypoints_2d: np.ndarray, width: int, height: int, *, K: tuple | None = None, height_m: float | None = None, floor_angle_deg: float | None = None, direction: str = 'side') -> np.ndarray:
    """Expose 3D point computation for TRC export/diagnostics."""
    return map_2d_to_3d(keypoints_2d, width, height, K=K, height_m=height_m, floor_angle_deg=floor_angle_deg, direction=direction)


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
    Build LOCAL bone transforms suitable for Live Link:
      - Root (head) Location = [0,0,0], Rotation = identity
      - Each child bone Location = [length,0,0] where length = |child-parent|
      - Rotation aligns +X with (child-parent)
    """
    # 3D points in UE space (approximate)
    points_3d = map_2d_to_3d(keypoints_2d, width, height, K=K, height_m=height_m, floor_angle_deg=floor_angle_deg, direction=direction)

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


