#!/usr/bin/env python3
"""
mujoco_rgb_depth_pub.py
=======================
对照线感知节点：MuJoCo RGB + Depth → SyncFrame（供 YOLOE 消费）

工作原理：
  1. 加载 MuJoCo 模型（只渲染）
  2. 订阅 /odom，同步无人机位姿
  3. 每帧做两路 EGL 离屏渲染：
       color_renderer  → RGB 图像 (H×W×3  uint8)
       depth_renderer  → 深度图  (H×W     float32, metres)
  4. 打包为 SyncFrame:
       sf.rgb      = sensor_msgs/Image (bgr8)
       sf.depth    = sensor_msgs/Image (32FC1, metres)
       sf.cam_pose = 相机世界系位姿 (Odometry)
       sf.cam_info = 相机内参
  5. 发布 /sync_frame_img（供 yoloe_seg_node.py 消费）

depth 编码说明：
  发布 32FC1（float32, 单位 m）。
  yoloe_seg config 对应设置：depth.encoding=32FC1, depth.scale=1.0
"""

import os
os.environ.setdefault('MUJOCO_GL', 'egl')   # 必须在 import mujoco 之前 | EGL=GPU直连，不走X11

import rospy
import numpy as np
import mujoco
import yaml
from threading import Lock

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from quadrotor_msgs.msg import SyncFrame
from std_msgs.msg import Header


def _mat2quat(R: np.ndarray):
    """旋转矩阵 (3×3) → 四元数 (w,x,y,z)。"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s;                 z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return float(w), float(x), float(y), float(z)


def _make_image_msg(arr: np.ndarray, stamp, encoding: str) -> Image:
    """将 numpy 数组转为 sensor_msgs/Image，不依赖 cv_bridge。"""
    msg = Image()
    msg.header.stamp = stamp
    msg.encoding     = encoding
    if arr.ndim == 2:
        msg.height, msg.width = arr.shape
        msg.step = arr.itemsize * msg.width
    else:
        msg.height, msg.width = arr.shape[:2]
        msg.step = arr.itemsize * arr.shape[2] * msg.width
    msg.is_bigendian = False
    msg.data = arr.tobytes()
    return msg


# ─────────────────────────────────────────────────────────────────────────────
class MuJoCoRGBDepthPublisher:

    def __init__(self):
        rospy.init_node('mujoco_rgb_depth_pub', anonymous=False)

        # ── 读取配置 ─────────────────────────────────────────────────────────
        model_path  = rospy.get_param('~model_path',
            '/home/lab/program/mujoco_ros1_docker/catkin_ws/src/drone_urdf/scene_drone.xml')
        config_path = rospy.get_param('~config_path',
            '/home/lab/program/mujoco_ros1_docker/catkin_ws/src/mujoco_perception/cfg/config.yaml')

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        cam_cfg = cfg.get('camera', {})
        rd_cfg  = cfg.get('rgb_depth', {})

        self.W            = int(cam_cfg.get('width',  424))
        self.H            = int(cam_cfg.get('height', 240))
        fovy_deg          = float(cam_cfg.get('fovy',  60.0))
        self._dmin        = float(cam_cfg.get('depth_min', 0.15))
        self._dmax        = float(cam_cfg.get('depth_max', 5.0))
        rate_hz           = float(cam_cfg.get('publish_hz', 10.0))

        sf_topic          = rd_cfg.get('syncframe_topic', '/sync_frame_img')

        # ── MuJoCo 模型（只渲染）────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'depth_cam')
        if self._cam_id == -1:
            rospy.logerr("[rgb_depth] 'depth_cam' not found in model.")
            rospy.signal_shutdown("depth_cam not found")
            return

        # ── 两个渲染器 ───────────────────────────────────────────────────────
        # Color renderer（默认模式 = RGB）
        self._color_renderer = mujoco.Renderer(self.model,
                                               height=self.H, width=self.W)

        # Depth renderer
        self._depth_renderer = mujoco.Renderer(self.model,
                                               height=self.H, width=self.W)
        self._depth_renderer.enable_depth_rendering()

        # ── 相机内参 ─────────────────────────────────────────────────────────
        fovy_rad   = np.deg2rad(fovy_deg)
        self.fy    = (self.H / 2.0) / np.tan(fovy_rad / 2.0)
        self.fx    = self.fy
        self.cx    = self.W / 2.0
        self.cy    = self.H / 2.0

        # ── 深度裁剪平面 ─────────────────────────────────────────────────────
        ext         = self.model.stat.extent
        self._znear = self.model.vis.map.znear * ext
        self._zfar  = self.model.vis.map.zfar  * ext

        # ── 位姿缓存 ─────────────────────────────────────────────────────────
        self._pose_lock     = Lock()
        self._pos           = np.zeros(3)
        self._quat          = np.array([1., 0., 0., 0.])
        self._pose_received = False

        # ── 预建相机内参消息 ─────────────────────────────────────────────────
        self._cam_info_msg = self._build_camera_info()

        # ── ROS 通讯 ─────────────────────────────────────────────────────────
        rospy.Subscriber('/odom', Odometry, self._odom_cb, queue_size=1)
        self._pub_sf   = rospy.Publisher(sf_topic,  SyncFrame,  queue_size=1)
        self._pub_info = rospy.Publisher('/rgb_depth/camera_info',
                                         CameraInfo, queue_size=1)
        # 可选独立话题（调试用）
        self._pub_rgb  = rospy.Publisher('/rgb_depth/image_rgb',  Image, queue_size=1)
        self._pub_dep  = rospy.Publisher('/rgb_depth/image_depth', Image, queue_size=1)

        self._rate = rospy.Rate(rate_hz)
        rospy.loginfo(
            f"[rgb_depth] Ready | {self.W}×{self.H} fx={self.fx:.1f} "
            f"range=[{self._dmin},{self._dmax}]m → {sf_topic} @ {rate_hz}Hz")

    # ──────────────────────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        with self._pose_lock:
            self._pos[:]  = [p.x, p.y, p.z]
            self._quat[:] = [q.w, q.x, q.y, q.z]
            self._pose_received = True

    # ──────────────────────────────────────────────────────────────────────────
    def _timer_cb(self, _event):
        with self._pose_lock:
            pose_received = self._pose_received
            pos  = self._pos.copy()
            quat = self._quat.copy()

        # 没有 /odom 时也先渲染默认模型位姿，至少能看见相机视角窗口。
        if pose_received:
            self.data.qpos[0:3] = pos
            self.data.qpos[3:7] = quat
        else:
            rospy.logwarn_throttle(2.0, "[rgb_depth] Waiting for /odom, rendering default pose.")

        mujoco.mj_forward(self.model, self.data)

        now = rospy.Time.now()

        # ── RGB 渲染 ─────────────────────────────────────────────────────────
        self._color_renderer.update_scene(self.data, camera=self._cam_id)
        rgb_buf = self._color_renderer.render()            # (H, W, 3) uint8 RGB

        # RGB → BGR (OpenCV / ROS bgr8 约定)
        bgr_buf = rgb_buf[:, :, ::-1].copy()

        # ── 深度渲染 ─────────────────────────────────────────────────────────
        self._depth_renderer.update_scene(self.data, camera=self._cam_id)
        depth_buf = self._depth_renderer.render()          # (H, W) float32

        zn, zf = self._znear, self._zfar
        with np.errstate(divide='ignore', invalid='ignore'):
            depth_m = zn * zf / np.where(
                depth_buf > 0, zf - depth_buf * (zf - zn), 1e-9)
        depth_buf_flipped = depth_m.copy()
        # 裁剪到有效范围（无效处设 0 → YOLOE 节点按 depth.min 过滤）
        depth_buf_flipped = np.where(
            (depth_buf_flipped > self._dmin) & (depth_buf_flipped < self._dmax),
            depth_buf_flipped, 0.0).astype(np.float32)

        # ── 构造 ROS Image 消息 ──────────────────────────────────────────────
        rgb_msg   = _make_image_msg(bgr_buf,           now, 'bgr8')
        depth_msg = _make_image_msg(depth_buf_flipped, now, '32FC1')
        rgb_msg.header.frame_id   = 'depth_cam'
        depth_msg.header.frame_id = 'depth_cam'

        # 可选独立发布（调试用）
        if self._pub_rgb.get_num_connections() > 0:
            self._pub_rgb.publish(rgb_msg)
        if self._pub_dep.get_num_connections() > 0:
            self._pub_dep.publish(depth_msg)

        # ── 相机位姿 ─────────────────────────────────────────────────────────
        cam_pos = self.data.cam_xpos[self._cam_id].copy()
        cam_mat = self.data.cam_xmat[self._cam_id].reshape(3, 3).copy()
        qw, qx, qy, qz = _mat2quat(cam_mat)

        # ── 打包 SyncFrame ───────────────────────────────────────────────────
        sf = SyncFrame()
        sf.header.stamp    = now
        sf.header.frame_id = 'world'

        sf.rgb   = rgb_msg
        sf.depth = depth_msg

        self._cam_info_msg.header.stamp = now
        sf.cam_info = self._cam_info_msg

        sf.cam_pose.header.stamp    = now
        sf.cam_pose.header.frame_id = 'world'
        sf.cam_pose.pose.pose.position.x = float(cam_pos[0])
        sf.cam_pose.pose.pose.position.y = float(cam_pos[1])
        sf.cam_pose.pose.pose.position.z = float(cam_pos[2])
        sf.cam_pose.pose.pose.orientation.w = qw
        sf.cam_pose.pose.pose.orientation.x = qx
        sf.cam_pose.pose.pose.orientation.y = qy
        sf.cam_pose.pose.pose.orientation.z = qz

        body_pos = pos if pose_received else self.data.qpos[0:3]
        body_quat = quat if pose_received else self.data.qpos[3:7]
        sf.body_odom.header.stamp    = now
        sf.body_odom.header.frame_id = 'world'
        sf.body_odom.pose.pose.position.x = float(body_pos[0])
        sf.body_odom.pose.pose.position.y = float(body_pos[1])
        sf.body_odom.pose.pose.position.z = float(body_pos[2])
        sf.body_odom.pose.pose.orientation.w = float(body_quat[0])
        sf.body_odom.pose.pose.orientation.x = float(body_quat[1])
        sf.body_odom.pose.pose.orientation.y = float(body_quat[2])
        sf.body_odom.pose.pose.orientation.z = float(body_quat[3])

        self._pub_sf.publish(sf)

        self._cam_info_msg.header.stamp = now
        self._pub_info.publish(self._cam_info_msg)

    # ──────────────────────────────────────────────────────────────────────────
    def _build_camera_info(self) -> CameraInfo:
        ci = CameraInfo()
        ci.header.frame_id  = 'depth_cam'
        ci.width            = self.W
        ci.height           = self.H
        ci.distortion_model = 'plumb_bob'
        ci.D = [0.0] * 5
        ci.K = [self.fx, 0.0, self.cx,
                0.0, self.fy, self.cy,
                0.0, 0.0, 1.0]
        ci.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.P = [self.fx, 0.0, self.cx, 0.0,
                0.0, self.fy, self.cy, 0.0,
                0.0, 0.0, 1.0, 0.0]
        return ci

    def run(self):
        while not rospy.is_shutdown():
            self._timer_cb(None)
            self._rate.sleep()


if __name__ == '__main__':
    try:
        node = MuJoCoRGBDepthPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
