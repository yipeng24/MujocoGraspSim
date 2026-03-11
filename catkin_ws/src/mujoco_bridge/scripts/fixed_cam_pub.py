#!/usr/bin/env python3
"""
fixed_cam_pub.py
================
从 MuJoCo 固定全景相机（overview_cam）渲染 RGB 图像并发布为 ROS Image，
用于 rosbag 录制场景视频。

该节点维护一个独立的 MuJoCo model/data 副本，通过订阅 /odom 和
/joint_states 保持与主仿真状态同步，然后用 MuJoCo offscreen renderer
渲染固定相机画面。

发布话题：
  /fixed_cam/image_raw    sensor_msgs/Image (rgb8, 默认 640x480, 10 Hz)
  /fixed_cam/camera_info  sensor_msgs/CameraInfo

参数：
  ~model_path   : MuJoCo XML 路径（默认与 mujoco_ros_bridge 相同）
  ~camera_name  : 相机名（默认 "overview_cam"，也可用 "maincam"）
  ~width        : 渲染宽度 (默认 640)
  ~height       : 渲染高度 (默认 480)
  ~fps          : 发布帧率 (默认 10.0)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import rospy
import numpy as np
import mujoco

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg    import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg    import Header

try:
    from cv_bridge import CvBridge
    _CV_BRIDGE = True
except ImportError:
    _CV_BRIDGE = False
    rospy.logwarn_once("[fixed_cam_pub] cv_bridge not available, using raw encoding")


class FixedCamPub:
    def __init__(self):
        rospy.init_node('fixed_cam_pub', anonymous=False)

        model_path  = rospy.get_param('~model_path',
            '/home/lab/program/mujoco_ros1_docker/catkin_ws/src/drone_urdf/scene_drone_payload.xml')
        cam_name    = rospy.get_param('~camera_name',  'overview_cam')
        self._width  = int(rospy.get_param('~width',  640))
        self._height = int(rospy.get_param('~height', 480))
        fps          = float(rospy.get_param('~fps',   10.0))
        ns           = rospy.get_param('~topic_prefix', '/fixed_cam')

        # ── MuJoCo model（独立副本，与 bridge 不共享） ───────────────────────
        self.model    = mujoco.MjModel.from_xml_path(model_path)
        self.data     = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, self._height, self._width)

        # 相机 ID
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if self._cam_id == -1:
            rospy.logwarn(f"[fixed_cam_pub] Camera '{cam_name}' not found, "
                          f"falling back to 'maincam'")
            self._cam_name = 'maincam'
        else:
            self._cam_name = cam_name
        rospy.loginfo(f"[fixed_cam_pub] Using camera '{self._cam_name}' "
                      f"(id={self._cam_id}), {self._width}x{self._height} @ {fps} Hz")

        # ── 关节地址查找 ─────────────────────────────────────────────────────
        fj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'base_free')
        self._fj_qadr = int(self.model.jnt_qposadr[fj_id]) if fj_id != -1 else 0

        arm_jnt_names = ['arm_pitch_joint', 'arm_pitch2_joint', 'arm_roll_joint']
        self._arm_qadr = []
        self._arm_names = arm_jnt_names
        for jname in arm_jnt_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self._arm_qadr.append(int(self.model.jnt_qposadr[jid]) if jid != -1 else -1)

        # ── ROS ─────────────────────────────────────────────────────────────
        self._bridge = CvBridge() if _CV_BRIDGE else None

        self._img_pub  = rospy.Publisher(f'{ns}/image_raw',    Image,      queue_size=1)
        self._info_pub = rospy.Publisher(f'{ns}/camera_info',  CameraInfo, queue_size=1)

        rospy.Subscriber('/odom',         Odometry,   self._odom_cb,   queue_size=1,
                         buff_size=2**20)
        rospy.Subscriber('/joint_states', JointState, self._joints_cb, queue_size=1)

        mujoco.mj_forward(self.model, self.data)

        self._timer = rospy.Timer(rospy.Duration(1.0 / fps), self._render_cb)
        rospy.loginfo(f"[fixed_cam_pub] Ready. Publishing to {ns}/image_raw")

    # ── 状态同步 ─────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        """将 /odom 里的位置和姿态写入本地 MjData.qpos（freejoint）。"""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        adr = self._fj_qadr
        self.data.qpos[adr:adr+3]   = [p.x, p.y, p.z]
        # MuJoCo freejoint quat 格式：w, x, y, z
        self.data.qpos[adr+3:adr+7] = [q.w, q.x, q.y, q.z]

    def _joints_cb(self, msg: JointState):
        """将 /joint_states 里的臂关节角写入本地 MjData.qpos。"""
        name_pos = dict(zip(msg.name, msg.position))
        for jname, qadr in zip(self._arm_names, self._arm_qadr):
            if qadr != -1 and jname in name_pos:
                self.data.qpos[qadr] = float(name_pos[jname])

    # ── 渲染 & 发布 ──────────────────────────────────────────────────────────

    def _render_cb(self, _event):
        """定时器回调：mj_forward → render → publish。"""
        try:
            mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, camera=self._cam_name)
            rgb = self.renderer.render()   # shape (H, W, 3), dtype uint8, RGB

            stamp = rospy.Time.now()

            # 发布 Image
            if _CV_BRIDGE and self._bridge is not None:
                img_msg = self._bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
            else:
                img_msg = Image()
                img_msg.encoding    = 'rgb8'
                img_msg.height      = rgb.shape[0]
                img_msg.width       = rgb.shape[1]
                img_msg.step        = rgb.shape[1] * 3
                img_msg.data        = rgb.tobytes()
                img_msg.is_bigendian = False
            img_msg.header.stamp    = stamp
            img_msg.header.frame_id = self._cam_name
            self._img_pub.publish(img_msg)

            # 发布 CameraInfo（基本参数）
            fovy_rad = float(self.model.cam_fovy[self._cam_id]) * np.pi / 180.0 \
                       if self._cam_id != -1 else (70.0 * np.pi / 180.0)
            fy = (self._height / 2.0) / np.tan(fovy_rad / 2.0)
            fx = fy
            cx = self._width  / 2.0
            cy = self._height / 2.0

            info = CameraInfo()
            info.header.stamp    = stamp
            info.header.frame_id = self._cam_name
            info.height          = self._height
            info.width           = self._width
            info.distortion_model = 'plumb_bob'
            info.D = [0.0] * 5
            info.K = [fx, 0, cx,  0, fy, cy,  0, 0, 1]
            info.R = [1, 0, 0,    0, 1, 0,    0, 0, 1]
            info.P = [fx, 0, cx, 0,  0, fy, cy, 0,  0, 0, 1, 0]
            self._info_pub.publish(info)

        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[fixed_cam_pub] render error: {e}")

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        FixedCamPub().run()
    except rospy.ROSInterruptException:
        pass
