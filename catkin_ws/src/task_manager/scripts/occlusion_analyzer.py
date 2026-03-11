#!/usr/bin/env python3
"""
occlusion_analyzer.py
=====================
遮挡分析节点：判断可移动障碍物是否遮挡目标的抓取接近路径。

算法：
  1. 接收 tgt_pcl + obs_pcl（世界系点云）
  2. 体素化两个点云
  3. 遮挡判断：obstacle 体素 ∩ (target 体素 ∪ 接近通道圆柱) ≠ ∅
  4. 聚类障碍物体素（scipy.ndimage.label）→ 各障碍物质心
  5. 清除位置 = 质心沿"远离目标"方向偏移 place_dist

发布：
  /task/occlusion_state           std_msgs/String   "clear" | "occluded"
  /task/target_pose               geometry_msgs/PoseStamped
  /task/obstacle_poses            geometry_msgs/PoseArray  (遮挡障碍物质心)
  /task/obstacle_remove_poses     geometry_msgs/PoseArray  (对应清除目标点)
"""

import os
import rospy
import numpy as np
import yaml
from threading import Lock

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PoseArray, Pose

try:
    from scipy.ndimage import label as nd_label
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    rospy.logwarn('[occlusion] scipy not available — clustering disabled')


def _pts_from_cloud(msg: PointCloud2) -> np.ndarray:
    """Extract (N,3) float32 array from PointCloud2."""
    gen = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
    pts = np.array(list(gen), dtype=np.float32)
    return pts if pts.ndim == 2 and pts.shape[1] == 3 else np.empty((0, 3), np.float32)


def _voxelize(pts: np.ndarray, voxel_size: float):
    """Return set of (ix, iy, iz) integer voxel keys."""
    if pts.shape[0] == 0:
        return set()
    idx = np.floor(pts / voxel_size).astype(np.int32)
    return set(map(tuple, idx))


class OcclusionAnalyzer:

    def __init__(self):
        rospy.init_node('occlusion_analyzer', anonymous=False)

        config_path = rospy.get_param('~config_path',
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'cfg', 'config.yaml'))
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        prefix    = rospy.get_param('~perception_prefix',
                                   cfg.get('perception_prefix', 'gt_seg'))
        occ_cfg   = cfg.get('occlusion', {})
        out_cfg   = cfg.get('output', {})

        self._voxel      = float(occ_cfg.get('voxel_size', 0.03))
        self._app_r      = float(occ_cfg.get('approach_radius', 0.15))
        self._app_h      = float(occ_cfg.get('approach_height', 0.35))
        self._min_pts    = int(occ_cfg.get('min_cluster_pts', 5))
        rate_hz          = float(occ_cfg.get('publish_hz', 5.0))
        self._place_dist = float(cfg.get('fsm', {}).get('place_dist', 0.6))

        tgt_topic = f'/{prefix}/tgt_pcl'
        obs_topic = f'/{prefix}/obs_pcl'
        rospy.loginfo(f'[occlusion] tgt={tgt_topic}  obs={obs_topic}')

        self._lock       = Lock()
        self._tgt_pts    = np.empty((0, 3), np.float32)
        self._obs_pts    = np.empty((0, 3), np.float32)
        self._tgt_fresh  = False
        self._obs_fresh  = False

        rospy.Subscriber(tgt_topic, PointCloud2, self._cb_tgt, queue_size=1)
        rospy.Subscriber(obs_topic, PointCloud2, self._cb_obs, queue_size=1)

        state_topic   = out_cfg.get('occlusion_state_topic',  '/task/occlusion_state')
        tpose_topic   = out_cfg.get('target_pose_topic',      '/task/target_pose')
        oposes_topic  = out_cfg.get('obstacle_poses_topic',   '/task/obstacle_poses')
        rposes_topic  = out_cfg.get('remove_poses_topic',     '/task/obstacle_remove_poses')

        self._pub_state  = rospy.Publisher(state_topic,  String,       queue_size=1, latch=True)
        self._pub_tpose  = rospy.Publisher(tpose_topic,  PoseStamped,  queue_size=1, latch=True)
        self._pub_oposes = rospy.Publisher(oposes_topic, PoseArray,    queue_size=1, latch=True)
        self._pub_rposes = rospy.Publisher(rposes_topic, PoseArray,    queue_size=1, latch=True)

        self._rate = rospy.Rate(rate_hz)
        rospy.loginfo('[occlusion] Ready')

    # ──────────────────────────────────────────────────────────────────────────
    def _cb_tgt(self, msg: PointCloud2):
        pts = _pts_from_cloud(msg)
        with self._lock:
            self._tgt_pts   = pts
            self._tgt_fresh = True

    def _cb_obs(self, msg: PointCloud2):
        pts = _pts_from_cloud(msg)
        with self._lock:
            self._obs_pts   = pts
            self._obs_fresh = True

    # ──────────────────────────────────────────────────────────────────────────
    def _analyze(self, _event):
        with self._lock:
            tgt_pts = self._tgt_pts.copy()
            obs_pts = self._obs_pts.copy()

        now = rospy.Time.now()

        # ── 目标质心 ────────────────────────────────────────────────────────
        if tgt_pts.shape[0] == 0:
            rospy.logwarn_throttle(2.0, '[occlusion] Waiting for target point cloud.')
            return   # 没有目标点云，等待感知
        tgt_centroid = tgt_pts.mean(axis=0)

        # 发布目标位姿
        tp = PoseStamped()
        tp.header.stamp    = now
        tp.header.frame_id = 'world'
        tp.pose.position.x = float(tgt_centroid[0])
        tp.pose.position.y = float(tgt_centroid[1])
        tp.pose.position.z = float(tgt_centroid[2])
        tp.pose.orientation.w = 1.0
        self._pub_tpose.publish(tp)

        # ── 无障碍物时直接发 clear ───────────────────────────────────────────
        if obs_pts.shape[0] < self._min_pts:
            self._pub_state.publish(String(data='clear'))
            self._pub_oposes.publish(PoseArray(header=tp.header))
            self._pub_rposes.publish(PoseArray(header=tp.header))
            return

        # ── 体素化 ──────────────────────────────────────────────────────────
        tgt_voxels = _voxelize(tgt_pts, self._voxel)
        obs_voxels = _voxelize(obs_pts, self._voxel)

        # 接近通道圆柱（目标正上方，半径 approach_radius，高度 approach_height）
        cyl_voxels = self._approach_cylinder_voxels(tgt_centroid)

        # 遮挡区域 = 目标体素 ∪ 圆柱体素
        blocked_voxels = tgt_voxels | cyl_voxels

        # ── 聚类障碍物体素 ─────────────────────────────────────────────────
        clusters = self._cluster_obs(obs_pts)   # list of (N_i,3) arrays

        occluding_centroids = []
        remove_centroids    = []

        for cl_pts in clusters:
            if cl_pts.shape[0] < self._min_pts:
                continue
            cl_voxels = _voxelize(cl_pts, self._voxel)
            if cl_voxels & blocked_voxels:    # 与遮挡区域有重叠
                c = cl_pts.mean(axis=0)
                occluding_centroids.append(c)
                # 清除点 = 质心沿远离目标方向偏移
                d = c[:2] - tgt_centroid[:2]
                norm = np.linalg.norm(d)
                if norm > 1e-3:
                    d = d / norm
                else:
                    d = np.array([1.0, 0.0])
                remove_xy = c[:2] + d * self._place_dist
                remove_centroids.append(np.array([remove_xy[0], remove_xy[1], c[2]]))

        state = 'occluded' if occluding_centroids else 'clear'
        self._pub_state.publish(String(data=state))

        # 发布障碍物位置和清除位置
        op = PoseArray(); op.header = tp.header
        rp = PoseArray(); rp.header = tp.header
        for c, r in zip(occluding_centroids, remove_centroids):
            op.poses.append(self._make_pose(c))
            rp.poses.append(self._make_pose(r))
        self._pub_oposes.publish(op)
        self._pub_rposes.publish(rp)

        rospy.logdebug_throttle(2.0, f'[occlusion] state={state} '
                                     f'n_obs_clusters={len(clusters)} '
                                     f'occluding={len(occluding_centroids)}')

    # ──────────────────────────────────────────────────────────────────────────
    def _approach_cylinder_voxels(self, tgt_centroid) -> set:
        """体素化目标上方的接近通道圆柱。"""
        v = self._voxel
        r = self._app_r
        h = self._app_h
        cx, cy, cz = tgt_centroid

        # 在 XY 平面上枚举半径内的体素，Z 范围为 [cz, cz+h]
        n_xy = int(np.ceil(r / v)) + 1
        n_z  = int(np.ceil(h / v)) + 1
        ix0  = int(np.floor(cx / v))
        iy0  = int(np.floor(cy / v))
        iz0  = int(np.floor(cz / v))

        voxels = set()
        for dix in range(-n_xy, n_xy + 1):
            for diy in range(-n_xy, n_xy + 1):
                wx = (ix0 + dix) * v + v / 2
                wy = (iy0 + diy) * v + v / 2
                if (wx - cx)**2 + (wy - cy)**2 <= r**2:
                    for diz in range(0, n_z + 1):
                        voxels.add((ix0 + dix, iy0 + diy, iz0 + diz))
        return voxels

    def _cluster_obs(self, obs_pts) -> list:
        """使用体素网格 + 连通标签聚类障碍物点云。"""
        if not _SCIPY_OK or obs_pts.shape[0] == 0:
            return [obs_pts]

        v = self._voxel
        idx = np.floor(obs_pts / v).astype(np.int32)

        # 归一化到非负坐标
        offset = idx.min(axis=0)
        idx -= offset
        shape  = idx.max(axis=0) + 1

        grid = np.zeros(shape, dtype=np.uint8)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

        struct = np.ones((3, 3, 3), dtype=np.uint8)   # 26-连通
        labeled, n_labels = nd_label(grid, structure=struct)

        clusters = []
        for lbl in range(1, n_labels + 1):
            vox_idx = np.argwhere(labeled == lbl)   # (M,3) voxel indices
            vox_idx += offset
            # 找属于该聚类的原始点
            pts_idx = np.ravel_multi_index(
                (idx[:, 0] + offset[0] - offset[0],
                 idx[:, 1] + offset[1] - offset[1],
                 idx[:, 2] + offset[2] - offset[2]),
                shape)
            # 简化：直接用体素中心作为该聚类的点
            cl_pts = (vox_idx.astype(np.float32) + 0.5) * v
            clusters.append(cl_pts)

        return clusters

    @staticmethod
    def _make_pose(xyz) -> Pose:
        p = Pose()
        p.position.x = float(xyz[0])
        p.position.y = float(xyz[1])
        p.position.z = float(xyz[2])
        p.orientation.w = 1.0
        return p

    def run(self):
        while not rospy.is_shutdown():
            self._analyze(None)
            self._rate.sleep()


if __name__ == '__main__':
    node = OcclusionAnalyzer()
    node.run()
