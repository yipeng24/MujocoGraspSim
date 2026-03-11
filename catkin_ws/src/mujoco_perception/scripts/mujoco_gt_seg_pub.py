#!/usr/bin/env python3
"""
mujoco_gt_seg_pub.py
====================
主线感知节点：MuJoCo GT Segmentation → 三分类点云 → SyncFrame

三分类逻辑：
  ① target  : config 中指定的目标 body (如 cup)
  ② movable : 所有 freejoint body（排除 target），或 config 中显式指定
  ③ static  : 其余（地板、桌子、柱子、机身）

发布：
  /gt_seg/tgt_pcl          目标物体点云 (世界系)
  /gt_seg/obs_pcl          可移动障碍物点云 (世界系) — 供 task_manager
  /gt_seg/env_pcl          静态环境点云   (世界系，可视化)
  /sync_frame_pcl_world    SyncFrame:
                             tgt_pcl  = 目标
                             env_pcl  = 静态 + 可移动（规划器绕行二者）
"""

import os
os.environ.setdefault('MUJOCO_GL', 'egl')   # 必须在 import mujoco 之前 | EGL=GPU直连，不走X11

import rospy
import numpy as np
import mujoco
import yaml
from threading import Lock

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo, Image
from quadrotor_msgs.msg import SyncFrame

_GEOM_TYPE = int(mujoco.mjtObj.mjOBJ_GEOM)
_FREE_TYPE  = int(mujoco.mjtJoint.mjJNT_FREE)


def _build_cloud(pts: np.ndarray, stamp, frame_id='world') -> PointCloud2:
    msg = PointCloud2()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.height          = 1
    msg.width           = pts.shape[0]
    msg.fields = [
        PointField('x', 0,  PointField.FLOAT32, 1),
        PointField('y', 4,  PointField.FLOAT32, 1),
        PointField('z', 8,  PointField.FLOAT32, 1),
    ]
    msg.is_bigendian = False
    msg.point_step   = 12
    msg.row_step     = 12 * msg.width
    msg.is_dense     = True
    msg.data         = pts.astype(np.float32).tobytes()
    return msg


def _make_image_msg(arr: np.ndarray, stamp, encoding: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.encoding = encoding
    if arr.ndim == 2:
        msg.height, msg.width = arr.shape
        msg.step = arr.itemsize * msg.width
    else:
        msg.height, msg.width = arr.shape[:2]
        msg.step = arr.itemsize * arr.shape[2] * msg.width
    msg.is_bigendian = False
    msg.data = arr.tobytes()
    return msg


def _mat2quat(R: np.ndarray):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s;  x = (R[2,1]-R[1,2])*s; y = (R[0,2]-R[2,0])*s; z = (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0*np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])
        w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0*np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])
        w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s = 2.0*np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])
        w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    return float(w), float(x), float(y), float(z)


class MuJoCoGTSegPublisher:

    def __init__(self):
        rospy.init_node('mujoco_gt_seg_pub', anonymous=False)

        model_path  = rospy.get_param('~model_path',
            '/home/lab/program/mujoco_ros1_docker/catkin_ws/src/drone_urdf/scene_drone.xml')
        config_path = rospy.get_param('~config_path',
            '/home/lab/program/mujoco_ros1_docker/catkin_ws/src/mujoco_perception/cfg/config.yaml')

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        cam_cfg = cfg.get('camera', {})
        gt_cfg  = cfg.get('gt_seg', {})

        self.W         = int(cam_cfg.get('width',  424))
        self.H         = int(cam_cfg.get('height', 240))
        fovy_deg       = float(cam_cfg.get('fovy',  60.0))
        self._dmin     = float(cam_cfg.get('depth_min', 0.15))
        self._dmax     = float(cam_cfg.get('depth_max', 5.0))
        rate_hz        = float(cam_cfg.get('publish_hz', 10.0))

        self._stride     = int(gt_cfg.get('stride', 2))
        self._en_vis     = bool(gt_cfg.get('en_vis', True))
        self._self_clear_radius = float(gt_cfg.get('self_clear_radius', 0.35))
        target_bodies    = gt_cfg.get('target_body_names', ['cup'])
        movable_names    = gt_cfg.get('movable_obstacle_names', [])
        tgt_topic        = gt_cfg.get('tgt_pcl_topic',   '/gt_seg/tgt_pcl')
        obs_topic        = gt_cfg.get('obs_pcl_topic',   '/gt_seg/obs_pcl')
        env_topic        = gt_cfg.get('env_pcl_topic',   '/gt_seg/env_pcl')
        sf_topic         = gt_cfg.get('syncframe_topic', '/sync_frame_pcl_world')
        rgb_topic        = gt_cfg.get('rgb_topic',       '/gt_seg/image_rgb')
        overlay_topic    = gt_cfg.get('overlay_topic',   '/gt_seg/overlay')

        # ── MuJoCo 模型（只渲染）────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'depth_cam')
        if self._cam_id == -1:
            rospy.logerr("[gt_seg] 'depth_cam' not found."); rospy.signal_shutdown(""); return

        # ── 三分类 geom ID 集合 ─────────────────────────────────────────────
        self._target_geom_ids  = self._collect_geom_ids_for_bodies(target_bodies)
        self._movable_geom_ids = self._collect_movable_geom_ids(movable_names, target_bodies)
        self._self_geom_ids    = self._collect_geom_ids_for_subtree('base_link')

        rospy.loginfo(f"[gt_seg] Target  geom IDs : {sorted(self._target_geom_ids)}")
        rospy.loginfo(f"[gt_seg] Movable geom IDs : {sorted(self._movable_geom_ids)}")
        rospy.loginfo(f"[gt_seg] Self    geom IDs : {sorted(self._self_geom_ids)}")

        # ── 渲染器 ───────────────────────────────────────────────────────────
        self._depth_renderer = mujoco.Renderer(self.model, height=self.H, width=self.W)
        self._depth_renderer.enable_depth_rendering()
        self._seg_renderer   = mujoco.Renderer(self.model, height=self.H, width=self.W)
        self._seg_renderer.enable_segmentation_rendering()
        self._color_renderer = mujoco.Renderer(self.model, height=self.H, width=self.W)

        # ── 相机内参 ─────────────────────────────────────────────────────────
        fovy_rad    = np.deg2rad(fovy_deg)
        self.fy     = (self.H / 2.0) / np.tan(fovy_rad / 2.0)
        self.fx     = self.fy
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        ext          = self.model.stat.extent
        self._znear  = self.model.vis.map.znear * ext
        self._zfar   = self.model.vis.map.zfar  * ext

        # ── 预计算像素网格 ───────────────────────────────────────────────────
        s  = max(1, self._stride)
        uu, vv       = np.meshgrid(np.arange(0, self.W, s, dtype=np.float32),
                                   np.arange(0, self.H, s, dtype=np.float32))
        self._uu     = uu;  self._vv = vv
        self._u_idx  = uu.astype(np.int32)
        self._v_idx  = vv.astype(np.int32)

        # ── 位姿缓存 ─────────────────────────────────────────────────────────
        self._pose_lock     = Lock()
        self._pos           = np.zeros(3)
        self._quat          = np.array([1., 0., 0., 0.])
        self._pose_received = False

        # ── ROS 通讯 ─────────────────────────────────────────────────────────
        rospy.Subscriber('/odom', Odometry, self._odom_cb, queue_size=1)
        self._pub_tgt  = rospy.Publisher(tgt_topic, PointCloud2, queue_size=1)
        self._pub_obs  = rospy.Publisher(obs_topic, PointCloud2, queue_size=1)
        self._pub_env  = rospy.Publisher(env_topic, PointCloud2, queue_size=1)
        self._pub_sf   = rospy.Publisher(sf_topic,  SyncFrame,   queue_size=1)
        self._pub_info = rospy.Publisher('/gt_seg/camera_info', CameraInfo, queue_size=1)
        self._pub_rgb  = rospy.Publisher(rgb_topic, Image, queue_size=1)
        self._pub_ovl  = rospy.Publisher(overlay_topic, Image, queue_size=1)
        self._cam_info_msg = self._build_camera_info()

        self._rate = rospy.Rate(rate_hz)
        rospy.loginfo(f"[gt_seg] Ready | {self.W}×{self.H} stride={s} "
                      f"tgt={len(self._target_geom_ids)} movable={len(self._movable_geom_ids)} "
                      f"→ {sf_topic} @ {rate_hz}Hz")

    # ──────────────────────────────────────────────────────────────────────────
    def _collect_geom_ids_for_bodies(self, body_names):
        ids = set()
        for name in body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid == -1:
                rospy.logwarn(f"[gt_seg] Body '{name}' not found."); continue
            for gid in range(self.model.ngeom):
                if self.model.geom_bodyid[gid] == bid:
                    ids.add(gid)
        return ids

    def _collect_movable_geom_ids(self, obstacle_names, target_names):
        """
        若 obstacle_names 非空：只收集指定 body 的 geom。
        若为空：自动检测所有 freejoint body（排除 target_names 中的 body）。
        """
        ids = set()
        if obstacle_names:
            return self._collect_geom_ids_for_bodies(obstacle_names)

        # auto-detect: all bodies with a FREE joint, excluding targets and the drone body
        for bid in range(self.model.nbody):
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ''
            if body_name in target_names or body_name == 'base_link':
                continue
            has_free = any(
                self.model.jnt_bodyid[jid] == bid and
                self.model.jnt_type[jid] == _FREE_TYPE
                for jid in range(self.model.njnt)
            )
            if not has_free:
                continue
            for gid in range(self.model.ngeom):
                if self.model.geom_bodyid[gid] == bid:
                    gname = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or str(gid)
                    rospy.loginfo(
                        f"[gt_seg] Auto obstacle: '{body_name}' → geom '{gname}' (id={gid})")
                    ids.add(gid)
        return ids

    def _collect_geom_ids_for_subtree(self, root_body_name):
        """Collect all geom ids attached to root body and its descendants."""
        root_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
        if root_bid == -1:
            rospy.logwarn(f"[gt_seg] Body subtree root '{root_body_name}' not found.")
            return set()

        subtree_bodies = set()
        for bid in range(self.model.nbody):
            cur = bid
            while cur >= 0:
                if cur == root_bid:
                    subtree_bodies.add(bid)
                    break
                parent = int(self.model.body_parentid[cur])
                if parent == cur:
                    break
                cur = parent

        geom_ids = set()
        for gid in range(self.model.ngeom):
            if int(self.model.geom_bodyid[gid]) in subtree_bodies:
                geom_ids.add(gid)
        return geom_ids

    # ──────────────────────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
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
        if pose_received:
            self.data.qpos[0:3] = pos
            self.data.qpos[3:7] = quat
        else:
            rospy.logwarn_throttle(2.0, "[gt_seg] Waiting for /odom, rendering default pose.")
        mujoco.mj_forward(self.model, self.data)

        # ── RGB 渲染（相机视角调试）──────────────────────────────────────────
        self._color_renderer.update_scene(self.data, camera=self._cam_id)
        rgb_buf = self._color_renderer.render()
        bgr_buf = rgb_buf[:, :, ::-1].copy()

        # ── 深度渲染 → 公制深度 ──────────────────────────────────────────────
        self._depth_renderer.update_scene(self.data, camera=self._cam_id)
        depth_buf = self._depth_renderer.render()
        zn, zf    = self._znear, self._zfar
        with np.errstate(divide='ignore', invalid='ignore'):
            depth_m = zn * zf / np.where(depth_buf > 0, zf - depth_buf*(zf-zn), 1e-9)

        # ── 分割渲染 → geom ID 图 ────────────────────────────────────────────
        self._seg_renderer.update_scene(self.data, camera=self._cam_id)
        seg      = self._seg_renderer.render()   # (H,W,3) int32
        # MuJoCo Python segmentation render returns object id in channel 0 and
        # object type in channel 1. Reading them in the opposite order causes
        # the target mask to stay empty even when the object is visible.
        seg_geom = seg[:, :, 0]
        seg_type = seg[:, :, 1]
        seg_disp = seg.copy()

        # ── 三分类掩码 ───────────────────────────────────────────────────────
        valid    = (depth_m > self._dmin) & (depth_m < self._dmax)
        is_geom  = (seg_type == _GEOM_TYPE)
        self_pix = np.zeros((self.H, self.W), dtype=bool)
        for gid in self._self_geom_ids:
            self_pix |= (is_geom & (seg_geom == gid))
        valid &= ~self_pix

        tgt_pix = np.zeros((self.H, self.W), dtype=bool)
        for gid in self._target_geom_ids:
            tgt_pix |= (is_geom & (seg_geom == gid))

        obs_pix = np.zeros((self.H, self.W), dtype=bool)
        for gid in self._movable_geom_ids:
            obs_pix |= (is_geom & (seg_geom == gid))

        tgt_mask = valid &  tgt_pix
        obs_mask = valid &  obs_pix & ~tgt_pix      # movable obstacles
        env_mask = valid & ~tgt_pix & ~obs_pix       # static env (vis only)
        # For planning → env includes movable obstacles (planner must avoid them)
        env_plan_mask = obs_mask | env_mask

        rospy.logdebug_throttle(
            2.0,
            '[gt_seg] pixels tgt=%d obs=%d env=%d',
            int(tgt_mask.sum()), int(obs_mask.sum()), int(env_mask.sum()))

        # ── 采样 ─────────────────────────────────────────────────────────────
        tgt_sel  = tgt_mask[self._v_idx, self._u_idx]
        obs_sel  = obs_mask[self._v_idx, self._u_idx]
        env_sel  = env_mask[self._v_idx, self._u_idx]
        envp_sel = env_plan_mask[self._v_idx, self._u_idx]

        cam_pos = self.data.cam_xpos[self._cam_id].copy()
        cam_mat = self.data.cam_xmat[self._cam_id].reshape(3, 3).copy()
        now     = rospy.Time.now()

        tgt_pts   = self._project_pts(tgt_sel,  depth_m, cam_pos, cam_mat)
        obs_pts   = self._project_pts(obs_sel,  depth_m, cam_pos, cam_mat)
        env_pts   = self._project_pts(env_sel,  depth_m, cam_pos, cam_mat)   # vis
        envp_pts  = self._project_pts(envp_sel, depth_m, cam_pos, cam_mat)   # planning

        env_pts  = self._clear_near_drone(env_pts, pos)
        envp_pts = self._clear_near_drone(envp_pts, pos)

        tgt_cloud  = _build_cloud(tgt_pts,  now)
        obs_cloud  = _build_cloud(obs_pts,  now)
        env_cloud  = _build_cloud(env_pts,  now)
        envp_cloud = _build_cloud(envp_pts, now)

        # ── 发布 ─────────────────────────────────────────────────────────────
        # obs_pcl 始终发布（供 task_manager 分析），不受 en_vis 控制
        self._pub_obs.publish(obs_cloud)
        if self._en_vis:
            self._pub_tgt.publish(tgt_cloud)
            self._pub_env.publish(env_cloud)

        if self._pub_rgb.get_num_connections() > 0:
            rgb_msg = _make_image_msg(bgr_buf, now, 'bgr8')
            rgb_msg.header.frame_id = 'depth_cam'
            self._pub_rgb.publish(rgb_msg)

        if self._pub_ovl.get_num_connections() > 0:
            overlay = bgr_buf.copy()
            tgt_disp = np.zeros((self.H, self.W), dtype=bool)
            for gid in self._target_geom_ids:
                tgt_disp |= ((seg_disp[:, :, 1] == _GEOM_TYPE) & (seg_disp[:, :, 0] == gid))
            obs_disp = np.zeros((self.H, self.W), dtype=bool)
            for gid in self._movable_geom_ids:
                obs_disp |= ((seg_disp[:, :, 1] == _GEOM_TYPE) & (seg_disp[:, :, 0] == gid))
            overlay[tgt_disp] = np.array([0, 0, 255], dtype=np.uint8)
            overlay[obs_disp & ~tgt_disp] = np.array([0, 255, 255], dtype=np.uint8)
            ovl_msg = _make_image_msg(overlay, now, 'bgr8')
            ovl_msg.header.frame_id = 'depth_cam'
            self._pub_ovl.publish(ovl_msg)

        self._cam_info_msg.header.stamp = now
        self._pub_info.publish(self._cam_info_msg)

        # SyncFrame: tgt vs (static+movable merged) → planning 会绕开可移动障碍物
        sf = self._build_syncframe(tgt_cloud, envp_cloud, pos, quat, cam_pos, cam_mat, now)
        self._pub_sf.publish(sf)

    # ──────────────────────────────────────────────────────────────────────────
    def _project_pts(self, sel_mask, depth_m, cam_pos, cam_mat) -> np.ndarray:
        if not np.any(sel_mask):
            return np.empty((0, 3), dtype=np.float32)
        d   = depth_m[self._v_idx, self._u_idx][sel_mask]
        u_s = self._uu[sel_mask];  v_s = self._vv[sel_mask]
        x_c =  (u_s - self.cx) / self.fx * d
        y_c = -(v_s - self.cy) / self.fy * d
        z_c = -d
        pts_world = (cam_mat @ np.stack([x_c, y_c, z_c])) + cam_pos[:, None]
        return pts_world.T.astype(np.float32)

    def _clear_near_drone(self, pts: np.ndarray, drone_pos) -> np.ndarray:
        if pts.shape[0] == 0 or self._self_clear_radius <= 0.0:
            return pts
        delta = pts - np.asarray(drone_pos, dtype=np.float32).reshape(1, 3)
        keep = np.einsum('ij,ij->i', delta, delta) >= self._self_clear_radius ** 2
        return pts[keep]

    def _build_syncframe(self, tgt_cloud, env_cloud, pos, quat,
                          cam_pos, cam_mat, stamp) -> SyncFrame:
        sf = SyncFrame()
        sf.header.stamp    = stamp;  sf.header.frame_id = 'world'
        sf.tgt_pcl = tgt_cloud;      sf.env_pcl = env_cloud

        sf.body_odom.header.stamp    = stamp
        sf.body_odom.header.frame_id = 'world'
        sf.body_odom.pose.pose.position.x    = float(pos[0])
        sf.body_odom.pose.pose.position.y    = float(pos[1])
        sf.body_odom.pose.pose.position.z    = float(pos[2])
        sf.body_odom.pose.pose.orientation.w = float(quat[0])
        sf.body_odom.pose.pose.orientation.x = float(quat[1])
        sf.body_odom.pose.pose.orientation.y = float(quat[2])
        sf.body_odom.pose.pose.orientation.z = float(quat[3])

        self._cam_info_msg.header.stamp = stamp
        sf.cam_info = self._cam_info_msg

        sf.cam_pose.header.stamp    = stamp
        sf.cam_pose.header.frame_id = 'world'
        sf.cam_pose.pose.pose.position.x = float(cam_pos[0])
        sf.cam_pose.pose.pose.position.y = float(cam_pos[1])
        sf.cam_pose.pose.pose.position.z = float(cam_pos[2])
        qw, qx, qy, qz = _mat2quat(cam_mat)
        sf.cam_pose.pose.pose.orientation.w = qw
        sf.cam_pose.pose.pose.orientation.x = qx
        sf.cam_pose.pose.pose.orientation.y = qy
        sf.cam_pose.pose.pose.orientation.z = qz
        return sf

    def _build_camera_info(self) -> CameraInfo:
        ci = CameraInfo()
        ci.header.frame_id  = 'depth_cam';  ci.width = self.W;  ci.height = self.H
        ci.distortion_model = 'plumb_bob';  ci.D = [0.0] * 5
        ci.K = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        ci.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.P = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return ci

    def run(self):
        while not rospy.is_shutdown():
            self._timer_cb(None)
            self._rate.sleep()


if __name__ == '__main__':
    node = MuJoCoGTSegPublisher()
    node.run()
