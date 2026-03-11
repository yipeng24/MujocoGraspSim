#!/usr/bin/env python3
"""
yoloe_seg_node.py
=================
对照线感知节点：YOLOE 开放词汇分割 → 三分类点云 → SyncFrame

文本提示分两类：
  target_prompts   → tgt_pcl（目标物体）
  obstacle_prompts → obs_pcl（可移动障碍物，供 task_manager）
  其余深度像素      → env_pcl（静态环境）

数据流：
  /sync_frame_img (SyncFrame: rgb bgr8 + depth 32FC1 + cam_pose)
      ↓
  YOLOE.predict(rgb)  → 实例分割 Masks + 类别索引
      ↓
  深度反投影 + cam_pose 变换 → 世界系点云
      ↓
  /yoloe_seg/tgt_pcl   (前景：目标物体)
  /yoloe_seg/obs_pcl   (可移动障碍物，供 task_manager)
  /yoloe_seg/env_pcl   (背景：静态环境)
  /sync_frame_out      (SyncFrame：tgt_pcl + env_pcl，供 bridge_node 打包)
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import rospy
import cv2
import torch
import numpy as np
import ros_numpy as rnp

from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from quadrotor_msgs.msg import SyncFrame

from data_manager import ConfigManager

# numpy 兼容性补丁
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'):   np.int   = int
if not hasattr(np, 'bool'):  np.bool  = bool


# ─────────────────────────────────────────────────────────────────────────────
def image_to_numpy(msg: Image, want_bgr=True) -> np.ndarray:
    arr = rnp.numpify(msg)
    enc = msg.encoding.lower() if msg.encoding else ""
    if arr.ndim == 3 and want_bgr and 'rgb' in enc and arr.shape[2] == 3:
        arr = arr[:, :, ::-1].copy()
    return arr


def numpy_to_image(arr: np.ndarray, header: Header, encoding: str) -> Image:
    msg = rnp.msgify(Image, arr, encoding=encoding)
    msg.header = header
    return msg


# ─────────────────────────────────────────────────────────────────────────────
class YOLOESegNode:

    def __init__(self):
        rospy.init_node('yoloe_seg', anonymous=False)

        # ── 配置 ─────────────────────────────────────────────────────────────
        config_path = rospy.get_param('~config_path',
            os.path.join(os.path.dirname(current_dir), 'cfg', 'config.yaml'))
        rospy.loginfo(f"[yoloe_seg] Loading config: {config_path}")
        self.cfg = ConfigManager(config_path)

        sync_topic        = self.cfg.get('sync_frame_topic', '/sync_frame_img')
        model_path        = self.cfg.get('model.path',  'yoloe-11l-seg.pt')
        self.text_prompts     = self.cfg.get('model.text_prompts', ['cup'])
        self.obstacle_prompts = self.cfg.get('model.obstacle_prompts', [])
        self.conf         = float(self.cfg.get('model.conf',  0.4))
        self.imgsz        = int(self.cfg.get('model.imgsz',   640))
        self.device       = self.cfg.get('model.device', 'auto')
        self.alpha        = float(self.cfg.get('model.alpha', 0.5))

        self.depth_encoding = self.cfg.get('depth.encoding', '32FC1')
        self.depth_scale    = float(self.cfg.get('depth.scale', 1.0))
        self.depth_min      = float(self.cfg.get('depth.min',   0.10))
        self.depth_max      = float(self.cfg.get('depth.max',   5.0))
        self.pt_stride      = int(self.cfg.get('point_cloud.stride', 2))
        self.publish_empty  = bool(self.cfg.get('point_cloud.publish_empty', True))
        self.en_vis         = bool(self.cfg.get('en_vis', True))

        tgt_topic    = self.cfg.get('output.tgt_pcl_topic', '/yoloe_seg/tgt_pcl')
        obs_topic    = self.cfg.get('output.obs_pcl_topic', '/yoloe_seg/obs_pcl')
        env_topic    = self.cfg.get('output.env_pcl_topic', '/yoloe_seg/env_pcl')
        sf_out_topic = self.cfg.get('output.syncframe_out', '/sync_frame_out')
        mask_topic   = self.cfg.get('output.mask_topic',    '/yoloe_seg/mask')
        ovrl_topic   = self.cfg.get('output.overlay_topic', '/yoloe_seg/overlay')

        default_K = [207.8, 0.0, 212.0, 0.0, 207.8, 120.0, 0.0, 0.0, 1.0]
        K_param   = self.cfg.get('camera.K', default_K)
        self.K    = np.array(K_param, dtype=np.float64).reshape(3, 3)

        # ── 加载 YOLOE 模型 ──────────────────────────────────────────────────
        # 合并 target + obstacle 提示词，用类别索引区分
        all_prompts = self.text_prompts + self.obstacle_prompts
        self._n_tgt = len(self.text_prompts)       # class 0..n_tgt-1 → target
        self._n_obs = len(self.obstacle_prompts)   # class n_tgt..end  → obstacle

        rospy.loginfo(f"[yoloe_seg] Loading YOLOE from: {model_path}")
        try:
            from ultralytics import YOLOE
            self.model = YOLOE(model_path)
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)

            rospy.loginfo(f"[yoloe_seg] Target prompts  : {self.text_prompts}")
            rospy.loginfo(f"[yoloe_seg] Obstacle prompts: {self.obstacle_prompts}")
            self.model.set_classes(all_prompts, self.model.get_text_pe(all_prompts))
            rospy.loginfo(f"[yoloe_seg] Model ready on: {self.device}")
        except Exception as e:
            rospy.logerr(f"[yoloe_seg] Model load failed: {e}")

        if self.device == 'cuda':
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        # ── ROS 通讯 ─────────────────────────────────────────────────────────
        self.pub_tgt     = rospy.Publisher(tgt_topic,    PointCloud2, queue_size=1)
        self.pub_obs     = rospy.Publisher(obs_topic,    PointCloud2, queue_size=1)
        self.pub_env     = rospy.Publisher(env_topic,    PointCloud2, queue_size=1)
        self.pub_sf_out  = rospy.Publisher(sf_out_topic, SyncFrame,   queue_size=1)
        self.pub_mask    = rospy.Publisher(mask_topic,   Image,       queue_size=1)
        self.pub_overlay = rospy.Publisher(ovrl_topic,   Image,       queue_size=1)
        self.pub_cam_pose = rospy.Publisher('/yoloe_seg/cam_pose',
                                            Odometry,    queue_size=1)

        rospy.Subscriber(sync_topic, SyncFrame, self.cb_syncframe,
                         queue_size=1, buff_size=2**24)
        rospy.loginfo(f"[yoloe_seg] Ready | sub={sync_topic} out={sf_out_topic}")

    # ──────────────────────────────────────────────────────────────────────────
    def cb_syncframe(self, sf: SyncFrame):
        # 1. 解码 RGB
        try:
            bgr = image_to_numpy(sf.rgb, want_bgr=True)
            if bgr.dtype != np.uint8:
                bgr = bgr.astype(np.uint8, copy=False)
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[yoloe_seg] RGB decode: {e}")
            return

        # 2. 解码深度（32FC1 = float32 metres）
        try:
            depth_np = rnp.numpify(sf.depth)
            enc = sf.depth.encoding if self.depth_encoding == 'auto' \
                  else self.depth_encoding
            if '16uc1' in enc.lower():
                depth_m = depth_np.astype(np.float32) * self.depth_scale
            else:
                depth_m = depth_np.astype(np.float32) * self.depth_scale
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[yoloe_seg] Depth decode: {e}")
            return

        h, w = bgr.shape[:2]
        if depth_m.shape[:2] != (h, w):
            depth_m = cv2.resize(depth_m, (w, h),
                                 interpolation=cv2.INTER_NEAREST)

        # 3. YOLOE 推理
        if not hasattr(self, 'model'):
            return
        try:
            results = self.model.predict(
                source=bgr, imgsz=self.imgsz, conf=self.conf,
                device=self.device, verbose=False)
            res = results[0]
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[yoloe_seg] Inference: {e}")
            return

        # 4. 构建前景/障碍物掩码（按类别索引分类）
        tgt_mask = np.zeros((h, w), dtype=np.uint8)
        obs_mask = np.zeros((h, w), dtype=np.uint8)
        try:
            if res.masks is not None and res.masks.data is not None:
                cls_ids = res.boxes.cls.cpu().numpy().astype(int) \
                          if res.boxes is not None else []
                for idx, m_tensor in enumerate(res.masks.data):
                    m = m_tensor.detach().cpu().numpy()
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.float32), (w, h)) > 0.5
                    else:
                        m = m > 0.5
                    cls_id = int(cls_ids[idx]) if idx < len(cls_ids) else 0
                    if cls_id < self._n_tgt:
                        tgt_mask[m] = 255
                    else:
                        obs_mask[m] = 255
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[yoloe_seg] Mask build: {e}")

        # 5. 发布可视化（叠加目标和障碍物掩码）
        vis_mask = np.maximum(tgt_mask, obs_mask)
        self._publish_vis(sf.header, bgr, tgt_mask, obs_mask)

        # 6. 生成分类点云并打包 SyncFrame
        self._process_point_cloud(sf, depth_m, tgt_mask, obs_mask, w, h)

    # ──────────────────────────────────────────────────────────────────────────
    def _process_point_cloud(self, sf, depth_m, tgt_mask, obs_mask, w, h):
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        valid = (np.isfinite(depth_m)
                 & (depth_m > self.depth_min)
                 & (depth_m < self.depth_max))

        stride = max(1, self.pt_stride)
        us = np.arange(0, w, stride, dtype=np.int32)
        vs = np.arange(0, h, stride, dtype=np.int32)
        uu, vv = np.meshgrid(us, vs)

        sel     = valid[vv, uu]
        is_tgt  = (tgt_mask[vv, uu] > 0)
        is_obs  = (obs_mask[vv, uu] > 0) & ~is_tgt
        tgt_sel = is_tgt & sel
        obs_sel = is_obs & sel
        bg_sel  = (~is_tgt) & (~is_obs) & sel

        world_frame_id = (sf.cam_pose.header.frame_id
                          if sf.cam_pose.header.frame_id else 'world')
        header_world = Header()
        header_world.stamp    = sf.header.stamp
        header_world.frame_id = world_frame_id

        if not np.any(sel):
            if self.publish_empty:
                empty = self._empty_cloud(header_world)
                if self.en_vis:
                    self.pub_tgt.publish(empty)
                    self.pub_env.publish(empty)
                self.pub_obs.publish(empty)
                self._pub_syncframe(sf, empty, empty)
            self._republish_pose(sf)
            return

        Z = depth_m[vv, uu]
        X = (uu - cx) * Z / fx
        Y = (vv - cy) * Z / fy

        def get_pts(selector):
            if not np.any(selector):
                return np.empty((0, 3), dtype=np.float32)
            return np.stack(
                [X[selector], Y[selector], Z[selector]], axis=1
            ).astype(np.float32)

        pts_tgt_cam = get_pts(tgt_sel)
        pts_obs_cam = get_pts(obs_sel)
        pts_bg_cam  = get_pts(bg_sel)

        pts_tgt_world = self._cam_to_world(pts_tgt_cam, sf.cam_pose)
        pts_obs_world = self._cam_to_world(pts_obs_cam, sf.cam_pose)
        pts_bg_world  = self._cam_to_world(pts_bg_cam,  sf.cam_pose)

        cloud_tgt = self._create_cloud(header_world, pts_tgt_world)
        cloud_obs = self._create_cloud(header_world, pts_obs_world)
        cloud_bg  = self._create_cloud(header_world, pts_bg_world)

        # obs_pcl 始终发布，供 task_manager
        self.pub_obs.publish(cloud_obs)
        if self.en_vis:
            self.pub_tgt.publish(cloud_tgt)
            self.pub_env.publish(cloud_bg)

        self._republish_pose(sf)
        # SyncFrame env_pcl = obs + env (规划器绕行二者)
        pts_env_plan = np.concatenate([pts_obs_world, pts_bg_world], axis=0) \
                       if pts_obs_world.shape[0] > 0 else pts_bg_world
        cloud_env_plan = self._create_cloud(header_world, pts_env_plan)
        self._pub_syncframe(sf, cloud_tgt, cloud_env_plan)

    # ──────────────────────────────────────────────────────────────────────────
    def _cam_to_world(self, pts_cam: np.ndarray, cam_pose) -> np.ndarray:
        if pts_cam.shape[0] == 0:
            return pts_cam
        o = cam_pose.pose.pose.orientation
        x, y, z, w = o.x, o.y, o.z, o.w
        R = np.array([
            [1 - 2*(y**2+z**2), 2*(x*y-z*w),      2*(x*z+y*w)],
            [2*(x*y+z*w),       1 - 2*(x**2+z**2), 2*(y*z-x*w)],
            [2*(x*z-y*w),       2*(y*z+x*w),       1 - 2*(x**2+y**2)]
        ], dtype=np.float32)
        p = cam_pose.pose.pose.position
        T = np.array([p.x, p.y, p.z], dtype=np.float32)
        return pts_cam @ R.T + T

    # ──────────────────────────────────────────────────────────────────────────
    def _pub_syncframe(self, sf, cloud_tgt, cloud_env):
        if self.pub_sf_out.get_num_connections() == 0:
            return
        out = SyncFrame()
        out.header   = sf.header
        out.rgb      = sf.rgb
        out.depth    = sf.depth
        out.tgt_pcl  = cloud_tgt
        out.env_pcl  = cloud_env
        out.cam_pose = sf.cam_pose
        out.cam_info = sf.cam_info
        self.pub_sf_out.publish(out)

    def _republish_pose(self, sf):
        if not self.en_vis:
            return
        if self.pub_cam_pose.get_num_connections() == 0:
            return
        odom = Odometry()
        odom.header.stamp    = sf.header.stamp
        odom.header.frame_id = sf.header.frame_id
        odom.pose            = sf.cam_pose.pose
        self.pub_cam_pose.publish(odom)

    def _publish_vis(self, header, bgr, tgt_mask, obs_mask):
        try:
            # 发布合并 mask：目标=白，障碍物=灰
            combined = np.zeros_like(tgt_mask)
            combined[obs_mask > 0] = 128
            combined[tgt_mask > 0] = 255
            self.pub_mask.publish(numpy_to_image(combined, header, 'mono8'))
            if self.pub_overlay.get_num_connections() > 0:
                overlay = bgr.copy()
                overlay[tgt_mask > 0, 2] = 255   # 目标 → 红通道
                overlay[obs_mask > 0, 0] = 255   # 障碍物 → 蓝通道
                overlay = cv2.addWeighted(bgr, 1.0, overlay, self.alpha, 0)
                self.pub_overlay.publish(
                    numpy_to_image(overlay, header, 'bgr8'))
        except Exception:
            pass

    def _create_cloud(self, header, pts: np.ndarray) -> PointCloud2:
        if pts.shape[0] == 0:
            return self._empty_cloud(header)
        fields = [PointField('x', 0, 7, 1),
                  PointField('y', 4, 7, 1),
                  PointField('z', 8, 7, 1)]
        return pc2.create_cloud(header, fields, pts.tolist())

    def _empty_cloud(self, header) -> PointCloud2:
        fields = [PointField('x', 0, 7, 1),
                  PointField('y', 4, 7, 1),
                  PointField('z', 8, 7, 1)]
        return pc2.create_cloud(header, fields, [])


if __name__ == '__main__':
    node = YOLOESegNode()
    rospy.spin()
