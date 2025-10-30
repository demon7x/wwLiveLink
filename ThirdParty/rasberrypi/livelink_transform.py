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


def map_2d_to_3d(points_2d: np.ndarray, width: int, height: int, scale: float = 0.1424) -> np.ndarray:
    """
    Map 2D image-space points to Unreal-like 3D local space.
    Current convention:
      - Origin at image center
      - X points right (scaled)
      - Y forward (0 for now)
      - Z up (0 for now)
    """
    origin_px = (width / 2.0, height / 2.0)
    pts = points_2d - np.array(origin_px)
    x_world = pts[:, 0] * scale
    y_world = np.zeros_like(x_world)
    z_world = np.zeros_like(x_world)
    return np.stack((x_world, y_world, z_world), axis=1)


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


def compute_transforms(keypoints_2d: np.ndarray, width: int, height: int, *, scale: float = 0.1424) -> list:
    points_3d = map_2d_to_3d(keypoints_2d, width, height, scale=scale)
    positions = get_bone_positions(points_3d)
    rotations = get_bone_rotations(points_3d)

    transforms = []
    for i, bone in enumerate(BONE_ORDER):
        transforms.append({
            "Location": positions[i],
            "Rotation": rotations[i],  # quaternion [x,y,z,w]
            "Scale": [1.0, 1.0, 1.0],
        })
    return transforms


