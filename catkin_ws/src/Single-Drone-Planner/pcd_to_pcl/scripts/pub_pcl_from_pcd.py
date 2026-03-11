#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pub_pcl_from_pcd.py
ROS1 node: Load a .pcd file with Open3D, optionally add a ground plane point cloud,
optionally apply a 3D rigid transform (rotation + translation), optionally remove
points inside/outside a vertical (z-axis) cylinder, then publish as sensor_msgs/PointCloud2.

Private params (~):
  # 基础
  ~pcd_path   (str)   : PCD 文件路径 (required)
  ~topic      (str)   : 发布 topic，默认 "/pointcloud"
  ~frame_id   (str)   : TF frame_id，默认 "map"
  ~hz         (int)   : 发布频率，默认 1
  ~latch      (bool)  : 是否 latched 发布，默认 True
  ~loop       (bool)  : 是否循环发布，默认 True（False=只发一次然后 spin）

  # 地面（可选）
  ~ground_enable (bool)  : 是否添加地面，默认 False
  ~ground_size   (list)  : [lx, ly] 地面矩形尺寸，默认 [5.0, 5.0]
  ~ground_z      (float) : 地面高度 z，默认 0.0
  ~ground_res    (float) : 地面点间距，默认 0.1（越小越密）
  ~ground_rgb    (list)  : 地面颜色 [r,g,b] 0-255，默认 [150,150,150]

  # 刚体变换（可选）
  ~transform_enable     (bool)  : 是否启用刚体变换，默认 False
  ~transform_xyz        (list)  : 平移 [tx, ty, tz]，默认 [0,0,0]
  ~transform_rpy_deg    (list)  : 欧拉角(度) [roll, pitch, yaw]，默认 [0,0,0]
  ~transform_quat_xyzw  (list)  : 四元数 [x,y,z,w]，若提供则优先于 rpy

  # 圆柱裁剪（可选）
  ~clip_cylinder_enable   (bool) : 是否启用圆柱裁剪，默认 False
  ~clip_cylinder_center   (list) : 圆柱中心 [cx, cy]，默认 [0.0, 0.0]
  ~clip_cylinder_radius   (float): 半径 R，默认 1.0
  ~clip_cylinder_zmin     (float): z 下界，默认 -inf
  ~clip_cylinder_zmax     (float): z 上界，默认 +inf
  ~clip_cylinder_keep_out (bool) : True=删除“内部点”保留外部；False=只保留内部点，默认 True

  # 手柄回字形点云（可选）
  ~hand_stick_pose_topic  (str) : 手柄位姿订阅 topic 名，默认 "hand_stick_pose"（可 remap）
  ~stick_ring_enable      (bool): 是否基于手柄位姿生成回字形点云，默认 False
  ~stick_ring_outer_size  (list): [ly, lz] 外方形尺寸（米），沿 y、z 轴，默认 [0.4, 0.4]
  ~stick_ring_inner_size  (list): [ly, lz] 内方形尺寸（米），沿 y、z 轴，默认 [0.2, 0.2]
  ~stick_ring_res         (float): 回字形点云分辨率（米），默认 0.02
  ~stick_ring_rgb         (list): 回字形颜色 [r,g,b] 0-255，默认 [255, 0, 0]
  ~stick_ring_offset_z    (float): 回字形 pose/点云在世界 z 方向移动的距离（米），默认 0.0
  ~stick_ring_pose_topic  (str) : 发布回字形位姿的 topic (geometry_msgs/PoseStamped)，默认 \"stick_ring_pose\"
"""

import math
import ast
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, PoseStamped
import sensor_msgs.point_cloud2 as pc2

try:
    import open3d as o3d
except ImportError as e:
    raise SystemExit(
        "[pcd_publisher] Open3D not found. Install with:\n"
        "  pip install open3d\n"
        f"Error: {e}"
    )

# ------------------------- Hand stick pose cache -------------------------
# Updated by subscriber; meant for other modules to read if needed.
hand_stick_pose = None  # ((x, y, z), (qx, qy, qz, qw))


def hand_stick_cb(msg: TransformStamped):
    global hand_stick_pose
    t = msg.transform.translation
    q = msg.transform.rotation
    hand_stick_pose = ((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

# ------------------------- Utils for robust param parsing -------------------------
def to_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
    return default

def to_float(v, default=0.0):
    if isinstance(v, (float, int, np.floating, np.integer)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("inf", "+inf", "infinity", "+infinity"):
            return float("inf")
        if s in ("-inf", "-infinity"):
            return float("-inf")
        try:
            return float(v)
        except Exception:
            pass
    return float(default)

def parse_list2(v, default=(0.0, 0.0)):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                return float(parsed[0]), float(parsed[1])
        except Exception:
            pass
    rospy.logwarn("Invalid 2-list param: %r, fallback to %r", v, default)
    return float(default[0]), float(default[1])


def parse_list3(v, default=(0.0, 0.0, 0.0)):
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return float(v[0]), float(v[1]), float(v[2])
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
                return float(parsed[0]), float(parsed[1]), float(parsed[2])
        except Exception:
            pass
    rospy.logwarn("Invalid 3-list param: %r, fallback to %r", v, default)
    return float(default[0]), float(default[1]), float(default[2])

def parse_center_xy(v, default=(0.0, 0.0)):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                return float(parsed[0]), float(parsed[1])
        except Exception:
            pass
    rospy.logwarn("~clip_cylinder_center is invalid: %r, fallback to %r", v, default)
    return float(default[0]), float(default[1])

# ------------------------- Rigid transform helpers -------------------------
def rpy_deg_to_rotmat(roll_deg, pitch_deg, yaw_deg):
    """Create rotation matrix using ZYX order: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    rd = math.radians(roll_deg)
    pd = math.radians(pitch_deg)
    yd = math.radians(yaw_deg)
    cr, sr = math.cos(rd), math.sin(rd)
    cp, sp = math.cos(pd), math.sin(pd)
    cy, sy = math.cos(yd), math.sin(yd)
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float64)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=np.float64)
    R = Rz @ Ry @ Rx
    return R.astype(np.float32)

def quat_xyzw_to_rotmat(x, y, z, w):
    """Quaternion (x,y,z,w) to rotation matrix."""
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = q / n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
    ], dtype=np.float64)
    return R.astype(np.float32)


def rotmat_to_quat_xyzw(R: np.ndarray):
    """Rotation matrix (3x3) -> quaternion (x, y, z, w)."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)

def apply_rigid_transform(points_xyz, t_xyz=(0.0,0.0,0.0), R=None):
    """
    Apply R @ p + t to all points.
    points_xyz: (N,3) float32
    t_xyz: (tx,ty,tz)
    R: (3,3) rotation matrix (float32). If None -> identity.
    """
    if points_xyz.size == 0:
        return points_xyz
    pts = points_xyz.astype(np.float32, copy=False)
    if R is not None:
        pts = pts @ R.T
    tx, ty, tz = float(t_xyz[0]), float(t_xyz[1]), float(t_xyz[2])
    if tx != 0.0 or ty != 0.0 or tz != 0.0:
        pts = pts + np.array([tx, ty, tz], dtype=np.float32)
    return pts

# ------------------------- Point cloud helpers -------------------------
def pack_rgb_to_float(rgb_uint8: np.ndarray) -> np.ndarray:
    rgb_uint32 = (rgb_uint8[:, 0].astype(np.uint32) << 16) | \
                 (rgb_uint8[:, 1].astype(np.uint32) << 8)  | \
                  rgb_uint8[:, 2].astype(np.uint32)
    return rgb_uint32.view(np.float32)

def rgb_tuple_to_float_array(n_pts: int, rgb_tuple):
    r, g, b = int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2])
    rgb_uint32 = (r << 16) | (g << 8) | b
    arr = np.full((n_pts,), rgb_uint32, dtype=np.uint32)
    return arr.view(np.float32)

def ensure_rgb(points_xyz: np.ndarray, rgb_f, default_rgb=(255,255,255)):
    """If rgb_f is None, create default white for all points."""
    if rgb_f is None:
        return rgb_tuple_to_float_array(points_xyz.shape[0], default_rgb)
    return rgb_f

def load_pcd(path: str):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"[pcd_publisher] Empty point cloud: {path}")
    pts = np.asarray(pcd.points, dtype=np.float32)
    rgb_f = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        colors = np.clip(colors, 0.0, 1.0)
        colors_uint8 = (colors * 255.0 + 0.5).astype(np.uint8)
        rgb_f = pack_rgb_to_float(colors_uint8)
    return pts, rgb_f

def generate_ground(size=(5.0, 5.0), z=0.0, res=0.1, rgb=(150,150,150)):
    """Generate a rectangular ground grid of points centered at (0,0,z)."""
    lx, ly = float(size[0]), float(size[1])
    res = max(float(res), 1e-4)  # avoid zero / too small
    # Use linspace to include edges
    nx = max(int(round(lx / res)) + 1, 2)
    ny = max(int(round(ly / res)) + 1, 2)
    xs = np.linspace(-lx/2.0, lx/2.0, nx, dtype=np.float32)
    ys = np.linspace(-ly/2.0, ly/2.0, ny, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys, indexing='xy')
    zc = np.full_like(xv, float(z), dtype=np.float32)
    pts = np.stack([xv.ravel(), yv.ravel(), zc.ravel()], axis=1).astype(np.float32)
    rgb_f = rgb_tuple_to_float_array(pts.shape[0], rgb)
    return pts, rgb_f

def filter_points_by_cylinder(points_xyz, rgb_f,
                              center_xy=(0.0, 0.0),
                              radius=1.0,
                              zmin=-math.inf,
                              zmax= math.inf,
                              keep_out=True):
    if points_xyz.size == 0:
        return points_xyz, rgb_f
    cx, cy = float(center_xy[0]), float(center_xy[1])
    R2 = float(radius) * float(radius)
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    inside_radial = (x - cx) * (x - cx) + (y - cy) * (y - cy) <= R2
    inside_height = (z >= zmin) & (z <= zmax)
    inside_cylinder = inside_radial & inside_height
    mask = ~inside_cylinder if keep_out else inside_cylinder
    pts_out = points_xyz[mask]
    rgb_out = rgb_f[mask] if rgb_f is not None else None
    return pts_out, rgb_out


def _make_plane_basis_from_line(p0, p1):
    """
    Create an orthonormal basis (u_dir, v_dir) for a plane that contains
    the line segment p0->p1 and is as vertical as possible (uses world Z as helper).
    """
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    u = p1 - p0
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-6:
        u_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        u_dir = u / u_norm

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    v = world_up - np.dot(world_up, u_dir) * u_dir  # project up onto plane
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-6:
        # line is almost vertical; pick an arbitrary horizontal axis
        alt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = alt - np.dot(alt, u_dir) * u_dir
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            v_norm = 1.0
    v_dir = v / v_norm
    return u_dir.astype(np.float32), v_dir.astype(np.float32)


def generate_square_ring(center, y_dir, z_dir,
                         outer_size=(0.4, 0.4),
                         inner_size=(0.2, 0.2),
                         res=0.02):
    """
    Generate a hollow square ring point cloud lying in the plane spanned by
    orthonormal axes:
      - y_dir : along p0->p1 (requested alignment)
      - z_dir : perpendicular to y_dir, as upward as possible
    (ring plane normal is x_dir = y_dir × z_dir).
    """
    res = max(float(res), 1e-4)
    outer_ly, outer_lz = float(outer_size[0]), float(outer_size[1])
    inner_ly, inner_lz = float(inner_size[0]), float(inner_size[1])

    # ensure inner is smaller than outer
    inner_ly = min(inner_ly, max(outer_ly - res, res))
    inner_lz = min(inner_lz, max(outer_lz - res, res))

    hy_out, hz_out = outer_ly * 0.5, outer_lz * 0.5
    hy_in, hz_in = inner_ly * 0.5, inner_lz * 0.5

    y_dir = np.asarray(y_dir, dtype=np.float32)
    z_dir = np.asarray(z_dir, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)

    ys = np.arange(-hy_out, hy_out + res * 0.5, res, dtype=np.float32)
    zs = np.arange(-hz_out, hz_out + res * 0.5, res, dtype=np.float32)
    yv, zv = np.meshgrid(ys, zs, indexing='xy')

    # keep outer area minus inner box
    mask = (np.abs(yv) >= hy_in) | (np.abs(zv) >= hz_in)
    yv = yv[mask]
    zv = zv[mask]
    if yv.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    pts = center + np.outer(yv, y_dir) + np.outer(zv, z_dir)
    return pts.astype(np.float32, copy=False)


# def make_pointcloud2(points_xyz: np.ndarray, rgb_f: np.ndarray, frame_id: str) -> PointCloud2:
#     header = rospy.Header()
#     header.stamp = rospy.Time.now()
#     header.frame_id = frame_id
#     if rgb_f is not None:
#         fields = [
#             PointField('x',   0,  PointField.FLOAT32, 1),
#             PointField('y',   4,  PointField.FLOAT32, 1),
#             PointField('z',   8,  PointField.FLOAT32, 1),
#             PointField('rgb', 12, PointField.FLOAT32, 1),
#         ]
#         data = np.concatenate([points_xyz, rgb_f.reshape(-1, 1)], axis=1)
#         gen = (tuple(row) for row in data)
#         cloud = pc2.create_cloud(header, fields, gen)
#     else:
#         cloud = pc2.create_cloud_xyz32(header, points_xyz)
#     return cloud


def make_pointcloud2(points_xyz: np.ndarray, rgb_f: np.ndarray, frame_id: str) -> PointCloud2:
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    if rgb_f is not None:
        fields = [
            PointField('x',   0,  PointField.FLOAT32, 1),
            PointField('y',   4,  PointField.FLOAT32, 1),
            PointField('z',   8,  PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1),
        ]
        # 确保 float32，避免 dtype 问题
        pts = points_xyz.astype(np.float32, copy=False)
        rgb = rgb_f.reshape(-1, 1).astype(np.float32, copy=False)
        data = np.concatenate([pts, rgb], axis=1)

        # 关键修改：生成“有长度的序列”，而不是生成器
        # 任选其一：
        points_seq = [tuple(row) for row in data]        # 方式1（推荐）
        # points_seq = data.tolist()                      # 方式2（更快但都是list而非tuple）

        cloud = pc2.create_cloud(header, fields, points_seq)
    else:
        # 纯XYZ，官方API本身会构造列表
        cloud = pc2.create_cloud_xyz32(header, points_xyz.astype(np.float32, copy=False))
    return cloud


# ------------------------- Main -------------------------
def main():
    rospy.init_node("pcd_publisher", anonymous=True)

    # 回字形点云参数
    stick_ring_enable = to_bool(rospy.get_param("~stick_ring_enable", False), False)
    ring_outer_param = rospy.get_param("~stick_ring_outer_size", [0.4, 0.4])
    ring_inner_param = rospy.get_param("~stick_ring_inner_size", [0.2, 0.2])
    ring_res = to_float(rospy.get_param("~stick_ring_res", 0.02), 0.02)
    ring_rgb_param = rospy.get_param("~stick_ring_rgb", [255, 0, 0])
    ring_pose_topic = rospy.get_param("~stick_ring_pose_topic", "stick_ring_pose")
    ring_offset_z_param = rospy.get_param("~stick_ring_offset_z", 0.0)
    ring_outer = parse_list2(ring_outer_param, (0.4, 0.4))
    ring_inner = parse_list2(ring_inner_param, (0.2, 0.2))
    ring_rgb_tuple = parse_list3(ring_rgb_param, (255, 0, 0))
    ring_offset_z = to_float(ring_offset_z_param, 0.0)

    # 仅在需要回字形时订阅手柄位姿
    if stick_ring_enable:
        hand_topic = rospy.get_param("~hand_stick_pose_topic", "hand_stick_pose")
        rospy.Subscriber(hand_topic, TransformStamped, hand_stick_cb, queue_size=1)

    # 基础参数
    pcd_path = rospy.get_param("~pcd_path", "")
    topic    = rospy.get_param("~topic", "/pointcloud")
    frame_id = rospy.get_param("~frame_id", "map")
    hz_param = rospy.get_param("~hz", 1)
    latch_p  = rospy.get_param("~latch", True)
    loop_p   = rospy.get_param("~loop", True)

    hz    = int(hz_param) if not isinstance(hz_param, str) else int(float(hz_param))
    latch = to_bool(latch_p, True)
    loop  = to_bool(loop_p, True)

    if not pcd_path:
        rospy.logerr("~pcd_path not set. Example:\n  rosrun your_pkg pub_pcl_from_pcd.py _pcd_path:=/path/to/cloud.pcd")
        return

    rospy.loginfo(f"[pcd_publisher] Loading PCD: {pcd_path}")
    points_xyz, rgb_f = load_pcd(pcd_path)
    print("points_xyz shape:", points_xyz.shape)
    print("points_xyz:", points_xyz)
    
    rospy.loginfo(f"[pcd_publisher] Loaded {points_xyz.shape[0]} points. Colors: {'yes' if rgb_f is not None else 'no'}")

    # 地面参数
    ground_enable = to_bool(rospy.get_param("~ground_enable", False), False)
    if ground_enable:
        size_param = rospy.get_param("~ground_size", [5.0, 5.0])
        ground_z   = to_float(rospy.get_param("~ground_z", 0.0), 0.0)
        res        = to_float(rospy.get_param("~ground_res", 0.1), 0.1)
        rgb_param  = rospy.get_param("~ground_rgb", [150, 150, 150])

        print("size_param:", size_param)

        # lx, ly = float(size_param[0]), float(size_param[1])
        # r, g, b = int(rgb_param[0]), int(rgb_param[1]), int(rgb_param[2])
        lx, ly = parse_list2(size_param, (0.0, 0.0))
        r, g, b = parse_list3(rgb_param, (0.0, 0, 0))

        g_pts, g_rgb = generate_ground((lx, ly), z=ground_z, res=res, rgb=(r, g, b))
        rospy.loginfo(f"[pcd_publisher] Ground generated: {g_pts.shape[0]} pts at z={ground_z}, size=({lx},{ly}), res={res}")

        # 如果原点云没有颜色，但地面有颜色，则给原点云补默认白色
        want_color = (rgb_f is not None) or (g_rgb is not None)
        if want_color:
            rgb_f = ensure_rgb(points_xyz, rgb_f, default_rgb=(255,255,255))
            points_xyz = np.vstack([points_xyz, g_pts])
            rgb_f = np.concatenate([rgb_f, g_rgb])
        else:
            points_xyz = np.vstack([points_xyz, g_pts])
            # rgb_f 仍为 None
    else:
        # no ground, keep original
        pass

    # 刚体变换参数（对合并后的点云一并处理）
    tf_enable   = to_bool(rospy.get_param("~transform_enable", False), False)
    print("tf_enable:", tf_enable)
    tf_enable = True
    t_xyz_param = rospy.get_param("~transform_xyz", [0.0, 0.0, 0.0])
    rpy_param   = rospy.get_param("~transform_rpy_deg", [0.0, 0.0, 0.0])
    quat_param  = rospy.get_param("~transform_quat_xyzw", None)

    tx, ty, tz = parse_list3(t_xyz_param, (0.0, 0.0, 0.0))
    rr, pp, yy = parse_list3(rpy_param, (0.0, 0.0, 0.0))

    R = None
    if tf_enable:
        if quat_param is not None:
            try:
                # parse quat [x,y,z,w]
                if isinstance(quat_param, (list, tuple)) and len(quat_param) == 4:
                    qx, qy, qz, qw = float(quat_param[0]), float(quat_param[1]), float(quat_param[2]), float(quat_param[3])
                elif isinstance(quat_param, str):
                    parsed = ast.literal_eval(quat_param)
                    qx, qy, qz, qw = float(parsed[0]), float(parsed[1]), float(parsed[2]), float(parsed[3])
                else:
                    raise ValueError("~transform_quat_xyzw must be length-4 [x,y,z,w].")
                R = quat_xyzw_to_rotmat(qx, qy, qz, qw)
                rospy.loginfo("[pcd_publisher] Using quaternion for rotation (x,y,z,w)=(%.4f, %.4f, %.4f, %.4f)", qx, qy, qz, qw)
            except Exception as e:
                rospy.logwarn("Invalid ~transform_quat_xyzw: %r, err=%s; fallback to RPY.", quat_param, str(e))
                R = rpy_deg_to_rotmat(rr, pp, yy)
        else:
            R = rpy_deg_to_rotmat(rr, pp, yy)

        before = points_xyz.copy()
        points_xyz = apply_rigid_transform(points_xyz, t_xyz=(tx, ty, tz), R=R)
        moved = np.linalg.norm(points_xyz - before, axis=1).mean() if before.size else 0.0
        rospy.loginfo("[pcd_publisher] Transform applied: t=(%.3f, %.3f, %.3f), rpy(deg)=(%.3f, %.3f, %.3f), mean|Δp|=%.4f",
                      tx, ty, tz, rr, pp, yy, moved)

    # 发布：基础点云固定，回字形根据手柄位置动态添加
    base_points = points_xyz
    base_rgb = rgb_f

    def build_cloud_with_ring():
        pts = base_points
        colors = base_rgb
        pose_msg = None

        if stick_ring_enable:
            if hand_stick_pose is None:
                rospy.logwarn_throttle(5.0, "[pcd_publisher] Waiting for hand stick pose; ring not added yet.")
            else:
                (px, py, pz), (qx, qy, qz, qw) = hand_stick_pose
                center = np.asarray([px, py, pz + ring_offset_z], dtype=np.float32)

                # Use hand stick orientation directly for ring axes.
                R = quat_xyzw_to_rotmat(qx, qy, qz, qw)
                y_dir = R[:, 1]
                z_dir = R[:, 2]

                ring_pts = generate_square_ring(center, y_dir, z_dir,
                                                outer_size=ring_outer,
                                                inner_size=ring_inner,
                                                res=ring_res)
                if ring_pts.size > 0:
                    if colors is not None:
                        ring_rgb = rgb_tuple_to_float_array(ring_pts.shape[0], ring_rgb_tuple)
                        pts = np.vstack([pts, ring_pts])
                        colors = np.concatenate([colors, ring_rgb])
                    else:
                        pts = np.vstack([pts, ring_pts])
                    # pose msg
                    pose_msg = PoseStamped()
                    pose_msg.header.frame_id = frame_id
                    pose_msg.pose.position.x = float(center[0])
                    pose_msg.pose.position.y = float(center[1])
                    pose_msg.pose.position.z = float(center[2])
                    pose_msg.pose.orientation.x = float(qx)
                    pose_msg.pose.orientation.y = float(qy)
                    pose_msg.pose.orientation.z = float(qz)
                    pose_msg.pose.orientation.w = float(qw)
                else:
                    rospy.logwarn_throttle(5.0, "[pcd_publisher] Ring generation returned zero points; check sizes/res.")

        return pts, colors, pose_msg

    pub = rospy.Publisher(topic, PointCloud2, queue_size=1, latch=latch)
    ring_pose_pub = rospy.Publisher(ring_pose_topic, PoseStamped, queue_size=1, latch=False) if stick_ring_enable else None
    rate = rospy.Rate(max(hz, 1))

    if loop:
        rospy.loginfo(f"[pcd_publisher] Publishing at {hz} Hz on {topic} (frame_id={frame_id}, latch={latch})")
        while not rospy.is_shutdown():
            pts_pub, rgb_pub, pose_msg = build_cloud_with_ring()
            msg = make_pointcloud2(pts_pub, rgb_pub, frame_id)
            msg.header.stamp = rospy.Time.now()
            pub.publish(msg)
            if ring_pose_pub is not None and pose_msg is not None:
                pose_msg.header.stamp = msg.header.stamp
                ring_pose_pub.publish(pose_msg)
            rate.sleep()
    else:
        rospy.loginfo(f"[pcd_publisher] Publishing once on {topic} (frame_id={frame_id}, latch={latch})")
        pts_pub, rgb_pub, pose_msg = build_cloud_with_ring()
        msg = make_pointcloud2(pts_pub, rgb_pub, frame_id)
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        if ring_pose_pub is not None and pose_msg is not None:
            pose_msg.header.stamp = msg.header.stamp
            ring_pose_pub.publish(pose_msg)
        rospy.loginfo("[pcd_publisher] Done. Spinning...")
        rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
