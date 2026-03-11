#!/usr/bin/env python3
"""
mujoco_depth_pub.py
===================
独立的 MuJoCo 深度相机点云发布节点。

工作原理：
  1. 加载同一份 MuJoCo 模型（scene_drone.xml），仅用于渲染，不做物理仿真
  2. 订阅 /odom → 将无人机位姿同步到本节点的 MjData
  3. 以指定频率（默认 10Hz）对 depth_cam 做离屏深度渲染（EGL）
  4. 深度图 → 世界坐标系点云 → 发布 /depth_cam/points

独立运行，不修改也不依赖 mujoco_ros_bridge.py。
用法：
  roslaunch mujoco_bridge mujoco_sim.launch use_viewer:=true use_depth:=true
"""

import os
# 强制 EGL 离屏渲染（不需要 DISPLAY）— 必须在 import mujoco 之前
os.environ.setdefault('MUJOCO_GL', 'egl')

import rospy
import numpy as np
import mujoco
from threading import Lock

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
from quadrotor_msgs.msg import SyncFrame


class MuJoCoDepthPublisher:
    def __init__(self):
        rospy.init_node('mujoco_depth_pub', anonymous=True)

        # ── MuJoCo 模型（只渲染，不仿真）──────────────────────────────────
        model_path = rospy.get_param(
            '~model_path',
            '/home/lambyeeh/program/mujoco_ros1_docker/catkin_ws/src/drone_urdf/scene_drone.xml')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # ── 无人机位姿（从 /odom 同步）───────────────────────────────────
        self._pose_lock     = Lock()
        self._pos           = np.array([0.0, 0.0, 0.0])
        self._quat          = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z
        self._pose_received = False

        # ── 查找 depth_cam ────────────────────────────────────────────────
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'depth_cam')
        if self._cam_id == -1:
            rospy.logerr("[depth_pub] 'depth_cam' not found in model. Check drone_converted.xml")
            rospy.signal_shutdown("depth_cam not found")
            return

        # ── 渲染分辨率 ────────────────────────────────────────────────────
        self.W = rospy.get_param('~width',  424)
        self.H = rospy.get_param('~height', 240)

        self._renderer = mujoco.Renderer(self.model, height=self.H, width=self.W)
        self._renderer.enable_depth_rendering()

        # ── 相机内参（与 XML 中 fovy=60° 一致）────────────────────────────
        fovy_deg = rospy.get_param('~fovy', 60.0)
        fovy_rad = np.deg2rad(fovy_deg)
        self.fy   = (self.H / 2.0) / np.tan(fovy_rad / 2.0)
        self.fx   = self.fy          # MuJoCo 正方形像素
        self.cx   = self.W / 2.0
        self.cy   = self.H / 2.0

        # ── 深度裁剪平面（与 MuJoCo 渲染一致）────────────────────────────
        ext           = self.model.stat.extent
        self._znear   = self.model.vis.map.znear * ext
        self._zfar    = self.model.vis.map.zfar  * ext
        self._dmin    = rospy.get_param('~depth_min', 0.15)   # 最近有效距离 m
        self._dmax    = rospy.get_param('~depth_max', 5.0)    # 最远有效距离 m

        # ── SyncFrame 发布（用于 planning 节点障碍物地图）────────────────
        self._pub_syncframe = rospy.get_param('~publish_syncframe', False)
        self._syncframe_topic = rospy.get_param(
            '~syncframe_topic', '/sync_frame_pcl_world')

        # ── ROS 订阅 / 发布 ───────────────────────────────────────────────
        rospy.Subscriber('/odom', Odometry, self._odom_cb, queue_size=1)
        self._cloud_pub   = rospy.Publisher(
            '/depth_cam/points',      PointCloud2, queue_size=1)
        self._caminfo_pub = rospy.Publisher(
            '/depth_cam/camera_info', CameraInfo,  queue_size=1)
        if self._pub_syncframe:
            self._sync_pub = rospy.Publisher(
                self._syncframe_topic, SyncFrame, queue_size=1)
            rospy.loginfo(
                f"[depth_pub] SyncFrame enabled → {self._syncframe_topic}")

        # ── 预计算像素网格（静态，节省每帧计算量）────────────────────────
        u_grid, v_grid = np.meshgrid(
            np.arange(self.W, dtype=np.float32),
            np.arange(self.H, dtype=np.float32))
        self._u_flat = u_grid.flatten()
        self._v_flat = v_grid.flatten()

        # ── 预构造 CameraInfo 消息（固定参数，只更新时间戳）──────────────
        self._cam_info_msg = self._build_camera_info()

        # ── 定时器：以固定频率发布 ────────────────────────────────────────
        rate_hz = rospy.get_param('~publish_hz', 10.0)
        rospy.Timer(rospy.Duration(1.0 / rate_hz), self._timer_cb)

        rospy.loginfo(
            f"[depth_pub] Ready | cam_id={self._cam_id}  {self.W}x{self.H}"
            f"  fx={self.fx:.1f}  znear={self._znear:.3f}m  zfar={self._zfar:.1f}m"
            f"  range=[{self._dmin}, {self._dmax}]m  {rate_hz}Hz")

    # ────────────────────────────────────────────────────────────────────
    # ROS 回调
    # ────────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        with self._pose_lock:
            self._pos[:]  = [p.x, p.y, p.z]
            self._quat[:] = [q.w, q.x, q.y, q.z]
            self._pose_received = True

    # ────────────────────────────────────────────────────────────────────
    # 内部方法
    # ────────────────────────────────────────────────────────────────────

    def _sync_drone_pose(self):
        """把最新 odometry 写入 MjData，更新运动学（相机位姿等）。"""
        with self._pose_lock:
            pos  = self._pos.copy()
            quat = self._quat.copy()
        # freejoint: qpos[0:3]=位置, qpos[3:7]=四元数(w,x,y,z)
        self.data.qpos[0:3] = pos
        self.data.qpos[3:7] = quat
        mujoco.mj_kinematics(self.model, self.data)  # 更新 cam_xpos / cam_xmat

    def _timer_cb(self, _event):
        """定时触发：渲染深度图并发布点云。"""
        if not self._pose_received:
            return

        self._sync_drone_pose()

        # ── 渲染深度图 ────────────────────────────────────────────────────
        self._renderer.update_scene(self.data, camera=self._cam_id)
        depth_buf = self._renderer.render()          # (H, W) float32, 值域 [0,1]

        # 归一化深度 → 公制深度（沿光轴方向距离，单位 m）
        zn, zf = self._znear, self._zfar
        with np.errstate(divide='ignore', invalid='ignore'):
            depth_m = zn * zf / np.where(
                depth_buf > 0,
                zf - depth_buf * (zf - zn),
                1e-9)

        # ── 像素 → 相机坐标系 3D 点 ──────────────────────────────────────
        # OpenGL 相机约定：X 右，Y 上，相机看向 -Z
        d = depth_m.flatten()
        mask = (d > self._dmin) & (d < self._dmax)
        d = d[mask]
        if d.size == 0:
            return

        x_c =  (self._u_flat[mask] - self.cx) / self.fx * d   # 右
        y_c = -(self._v_flat[mask] - self.cy) / self.fy * d   # 上（翻转图像 Y）
        z_c = -d                                               # 前方 → -Z

        pts_cam = np.stack([x_c, y_c, z_c], axis=1)           # (N, 3)

        # ── 相机坐标系 → 世界坐标系 ──────────────────────────────────────
        # cam_xmat 列向量：相机 X/Y/Z 轴在世界系下的方向
        cam_pos = self.data.cam_xpos[self._cam_id].copy()      # (3,)
        cam_mat = self.data.cam_xmat[self._cam_id].reshape(3, 3).copy()
        pts_world = (cam_mat @ pts_cam.T).T + cam_pos          # (N, 3)

        # ── 发布 ──────────────────────────────────────────────────────────
        now = rospy.Time.now()
        cloud_msg = self._build_cloud(pts_world, now)
        self._cloud_pub.publish(cloud_msg)
        self._cam_info_msg.header.stamp = now
        self._caminfo_pub.publish(self._cam_info_msg)

        # ── 发布 SyncFrame（供 planning 节点障碍物地图使用）──────────────
        if self._pub_syncframe:
            with self._pose_lock:
                pos  = self._pos.copy()
                quat = self._quat.copy()   # w,x,y,z
            sf = SyncFrame()
            sf.header.stamp    = now
            sf.header.frame_id = 'world'
            # env_pcl = 障碍物点云（规划使用）
            sf.env_pcl = cloud_msg
            # tgt_pcl = 空点云（无目标物体）
            sf.tgt_pcl = self._build_cloud(
                np.zeros((0, 3), dtype=np.float32), now)
            # body_odom = 无人机当前位姿
            sf.body_odom.header.stamp    = now
            sf.body_odom.header.frame_id = 'world'
            sf.body_odom.pose.pose.position.x    = float(pos[0])
            sf.body_odom.pose.pose.position.y    = float(pos[1])
            sf.body_odom.pose.pose.position.z    = float(pos[2])
            sf.body_odom.pose.pose.orientation.w = float(quat[0])
            sf.body_odom.pose.pose.orientation.x = float(quat[1])
            sf.body_odom.pose.pose.orientation.y = float(quat[2])
            sf.body_odom.pose.pose.orientation.z = float(quat[3])
            # cam_info = 相机内参
            self._cam_info_msg.header.stamp = now
            sf.cam_info = self._cam_info_msg
            # cam_pose = 相机在世界系下的位姿（Odometry 格式）
            sf.cam_pose.header.stamp    = now
            sf.cam_pose.header.frame_id = 'world'
            sf.cam_pose.pose.pose.position.x = float(cam_pos[0])
            sf.cam_pose.pose.pose.position.y = float(cam_pos[1])
            sf.cam_pose.pose.pose.position.z = float(cam_pos[2])
            # 旋转矩阵 → 四元数（Shepperd 方法）
            R = cam_mat
            trace = R[0,0] + R[1,1] + R[2,2]
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                cw = 0.25/s; cx = (R[2,1]-R[1,2])*s; cy = (R[0,2]-R[2,0])*s; cz = (R[1,0]-R[0,1])*s
            elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                cw = (R[2,1]-R[1,2])/s; cx = 0.25*s; cy = (R[0,1]+R[1,0])/s; cz = (R[0,2]+R[2,0])/s
            elif R[1,1] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                cw = (R[0,2]-R[2,0])/s; cx = (R[0,1]+R[1,0])/s; cy = 0.25*s; cz = (R[1,2]+R[2,1])/s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                cw = (R[1,0]-R[0,1])/s; cx = (R[0,2]+R[2,0])/s; cy = (R[1,2]+R[2,1])/s; cz = 0.25*s
            sf.cam_pose.pose.pose.orientation.w = float(cw)
            sf.cam_pose.pose.pose.orientation.x = float(cx)
            sf.cam_pose.pose.pose.orientation.y = float(cy)
            sf.cam_pose.pose.pose.orientation.z = float(cz)
            self._sync_pub.publish(sf)

    def _build_cloud(self, pts: np.ndarray, stamp) -> PointCloud2:
        msg = PointCloud2()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'world'
        msg.height    = 1
        msg.width     = pts.shape[0]
        msg.fields    = [
            PointField('x', 0,  PointField.FLOAT32, 1),
            PointField('y', 4,  PointField.FLOAT32, 1),
            PointField('z', 8,  PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step   = 12          # 3 × float32
        msg.row_step     = 12 * msg.width
        msg.is_dense     = True
        msg.data         = pts.astype(np.float32).tobytes()
        return msg

    def _build_camera_info(self) -> CameraInfo:
        ci = CameraInfo()
        ci.header.frame_id  = 'depth_cam'
        ci.width            = self.W
        ci.height           = self.H
        ci.distortion_model = 'plumb_bob'
        ci.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        ci.K = [self.fx, 0.0,     self.cx,
                0.0,     self.fy, self.cy,
                0.0,     0.0,     1.0]
        ci.R = [1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0]
        ci.P = [self.fx, 0.0,     self.cx, 0.0,
                0.0,     self.fy, self.cy, 0.0,
                0.0,     0.0,     1.0,     0.0]
        return ci


if __name__ == '__main__':
    try:
        node = MuJoCoDepthPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
