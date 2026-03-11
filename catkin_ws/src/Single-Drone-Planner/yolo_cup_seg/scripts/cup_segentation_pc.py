#/home/jr/anaconda3/envs/airgrasp/bin/python
# -*- coding: utf-8 -*-
import rospy
import sys
import os

# -------------------------------------------------------------------
# 1. 动态添加当前目录到 sys.path，确保能 import data_manager
# -------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import cv2
import torch
from ultralytics import YOLO
import ros_numpy as rnp
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from quadrotor_msgs.msg import SyncFrame
import numpy as np

# 导入配置管理器 (假设 data_manager.py 还在脚本同级目录)
from data_manager import ConfigManager

# 兼容性处理
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"):   np.int   = int
if not hasattr(np, "bool"):  np.bool  = bool

def image_to_numpy(msg: Image, want_bgr=True) -> np.ndarray:
    arr = rnp.numpify(msg)
    enc = msg.encoding.lower() if msg.encoding else ""
    if arr.ndim == 3:
        if want_bgr:
            if "rgb" in enc and arr.shape[2] == 3:
                arr = arr[:, :, ::-1].copy()
        else:
            if "bgr" in enc and arr.shape[2] == 3:
                arr = arr[:, :, ::-1].copy()
    return arr

def numpy_to_image(arr: np.ndarray, header: Header, encoding: str) -> Image:
    msg = rnp.msgify(Image, arr, encoding=encoding)
    msg.header = header
    return msg

class YOLO11SegFromSyncFramePack:
    def __init__(self):
        # -----------------------------------------------------------
        # 2. 修改配置路径：指向上一级目录下的 cfg/config.yaml
        # -----------------------------------------------------------
        # os.path.dirname(current_dir) 获取当前目录的父目录
        parent_dir = os.path.dirname(current_dir)
        config_file = os.path.join(parent_dir, "cfg", "config.yaml")
        
        # 打印一下路径方便调试
        rospy.loginfo(f"[PackNode] Looking for config at: {config_file}")
        
        self.cfg = ConfigManager(config_file)

        # -------- 读取参数 --------
        self.sync_frame_topic = self.cfg.get("sync_frame_topic", "/sync_frame_img")
        
        self.model_path = self.cfg.get("model.path", "")
        self.conf   = float(self.cfg.get("model.conf", 0.5))
        self.imgsz  = int(self.cfg.get("model.imgsz", 640))
        self.device = self.cfg.get("model.device", "auto")
        self.alpha  = float(self.cfg.get("model.alpha", 0.5))

        self.depth_encoding = self.cfg.get("depth.encoding", "auto")
        self.depth_scale    = float(self.cfg.get("depth.scale", 0.001))
        self.depth_min      = float(self.cfg.get("depth.min", 0.1))
        self.depth_max      = float(self.cfg.get("depth.max", 6.0))
        self.pt_stride      = int(self.cfg.get("point_cloud.stride", 2))
        self.publish_empty_clouds = bool(self.cfg.get("point_cloud.publish_empty", True))
        
        # 控制是否发布点云话题（独立的 /sync_tgt_pcl 和 /sync_env_pcl）
        self.en_vis = bool(self.cfg.get("en_vis", False))

        fg_ids_param = self.cfg.get("point_cloud.fg_class_ids", "auto")
        self.fg_class_ids = self._parse_fg_ids(fg_ids_param)

        default_K = [607.225, 0.0, 315.675, 0.0, 607.510, 233.738, 0.0, 0.0, 1.0]
        K_param = self.cfg.get("camera.K", default_K)
        self.K = np.array(K_param, dtype=np.float64).reshape(3, 3)

        # -------- 初始化模型 --------
        rospy.loginfo(f"[PackNode] Loading model from: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            rospy.loginfo(f"[PackNode] Model loaded on device: {self.device}")
        except Exception as e:
            rospy.logerr(f"[PackNode] Model load failed: {e}")

        if self.device == "cuda":
            try: torch.backends.cudnn.benchmark = True
            except: pass

        # -------- 初始化 ROS 通讯 --------
        self.pub_mask      = rospy.Publisher("~mask", Image, queue_size=1)
        self.pub_overlay   = rospy.Publisher("~overlay", Image, queue_size=1)
        self.pub_fg_cloud  = rospy.Publisher("/sync_tgt_pcl", PointCloud2, queue_size=1)
        self.pub_bg_cloud  = rospy.Publisher("/sync_env_pcl", PointCloud2, queue_size=1)
        self.pub_cam_pose  = rospy.Publisher("/sync_cam_pose_out", Odometry, queue_size=1)
        self.pub_sync_out  = rospy.Publisher("/sync_frame_out", SyncFrame, queue_size=1)

        self.sub_sync = rospy.Subscriber(self.sync_frame_topic, SyncFrame, self.cb_syncframe,
                                         queue_size=1, buff_size=2**24)

        rospy.loginfo(f"[PackNode] Ready. Topic: {self.sync_frame_topic}")

    def _parse_fg_ids(self, param):
        if str(param).lower() in ("auto", "", "none"):
            return None
        try:
            if isinstance(param, list): return [int(x) for x in param]
            return [int(x) for x in str(param).split(",")]
        except:
            return None

    def cb_syncframe(self, sf: SyncFrame):
        # 1. RGB Decode
        try:
            bgr = image_to_numpy(sf.rgb, want_bgr=True)
            if bgr.dtype != np.uint8: bgr = bgr.astype(np.uint8, copy=False)
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"RGB decode error: {e}")
            return

        # 2. Depth Decode
        try:
            depth_np = rnp.numpify(sf.depth)
            enc = sf.depth.encoding if self.depth_encoding == "auto" else self.depth_encoding
            if "16uc1" in enc.lower():
                depth_m = depth_np.astype(np.float32) * self.depth_scale
            else:
                depth_m = depth_np.astype(np.float32, copy=False)
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"Depth decode error: {e}")
            return

        h, w = bgr.shape[:2]
        if depth_m.shape[:2] != (h, w):
            depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

        # 3. Predict
        if not hasattr(self, 'model'): return
        try:
            res = self.model.predict(source=bgr, imgsz=self.imgsz, conf=self.conf,
                                     device=self.device, verbose=False)[0]
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"Inference error: {e}")
            return

        # 4. Generate Mask
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        try:
            if res.masks is not None and res.masks.data is not None:
                mdata = res.masks.data
                cls_ids = res.boxes.cls.detach().cpu().numpy().astype(int) if res.boxes else None
                n_classes = len(self.model.names) if hasattr(self.model, "names") else 1
                
                valid_ids = self.fg_class_ids if self.fg_class_ids is not None else (
                    [0] if n_classes <=1 else list(range(n_classes))
                )

                for i, m_tensor in enumerate(mdata):
                    if valid_ids is not None and cls_ids is not None:
                        if int(cls_ids[i]) not in valid_ids:
                            continue
                    
                    m = m_tensor.detach().cpu().numpy()
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.float32), (w, h)) > 0.5
                    else:
                        m = m > 0.5
                    fg_mask[m] = 255
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"Mask build error: {e}")

        self._publish_vis(sf.header, bgr, fg_mask)

        # 5. Point Cloud Generation
        self._process_point_cloud(sf, depth_m, fg_mask, w, h)

    # --------------------------------------------------------
    #   新增: 坐标变换辅助函数
    # --------------------------------------------------------
    def _get_rotation_matrix(self, orientation):
        """将四元数转换为旋转矩阵 (3x3)"""
        x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
        # 标准四元数转矩阵公式
        R = np.array([
            [1 - 2*(y**2 + z**2),  2*(x*y - z*w),      2*(x*z + y*w)],
            [2*(x*y + z*w),        1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),        2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)
        return R

    def _transform_pts_to_world(self, pts_cam, odom):
        """
        将相机系点云 (Nx3) 转换到世界系
        P_world = R * P_cam + T
        """
        if pts_cam.shape[0] == 0:
            return pts_cam

        # 1. 获取旋转矩阵 R (3x3)
        R = self._get_rotation_matrix(odom.pose.pose.orientation)
        
        # 2. 获取平移向量 T (3,)
        pos = odom.pose.pose.position
        T = np.array([pos.x, pos.y, pos.z], dtype=np.float32)

        # 3. 执行变换: pts_cam (N,3) -> pts_world (N,3)
        # 公式: P_world = R * P_cam + T
        # 由于 pts_cam 是行向量存储 (N,3)，所以写成: P @ R.T + T
        pts_world = np.dot(pts_cam, R.T) + T
        return pts_world

    def _refine_fg_bg_clouds(self, pts_fg, pts_bg, nb_neighbors=20, std_ratio=1.0, padding=0.01, bbox_z_offset=0.0):
            """
            1. 对 pts_fg 进行统计滤波去除离群点。
            2. 计算保留下来的 pts_fg 的 Bounding Box，并根据 bbox_z_offset 进行 Z 轴平移。
            3. 检查 pts_bg 中是否有位于该 Bounding Box 内的点。
            4. 将这些点从 pts_bg 移动到 pts_fg。
            
            :param nb_neighbors: 统计滤波计算邻居的数量
            :param std_ratio: 标准差倍数
            :param padding: Bounding Box 的额外外扩边距 (单位: 米)
            :param bbox_z_offset: Bounding Box 在 Z 轴方向的偏移量 (单位: 米)。
                                > 0 往上移 (远离地面/桌面), < 0 往下移。
            """
            # 如果 FG 点太少，无法进行统计滤波，直接返回
            if pts_fg.shape[0] < nb_neighbors:
                return pts_fg, pts_bg

            # --- 步骤 1: 统计滤波移除 FG 离群点 ---
            try:
                from scipy.spatial import cKDTree
            except ImportError:
                rospy.logerr("Scipy not installed. Skipping outlier removal.")
                return pts_fg, pts_bg

            tree = cKDTree(pts_fg)
            dists, _ = tree.query(pts_fg, k=nb_neighbors + 1)
            mean_dists = np.mean(dists[:, 1:], axis=1)
            
            global_mean = np.mean(mean_dists)
            global_std  = np.std(mean_dists)
            
            threshold = global_mean + std_ratio * global_std
            inlier_mask = mean_dists < threshold
            
            pts_fg_clean = pts_fg[inlier_mask]
            
            if pts_fg_clean.shape[0] == 0:
                rospy.logwarn("Outlier removal removed all points, reverting...")
                pts_fg_clean = pts_fg

            # --- 步骤 2: 基于偏移后的 Bounding Box 从 BG 召回点 ---
            if pts_fg_clean.shape[0] > 0:
                # 原始包围盒
                min_pt = np.min(pts_fg_clean, axis=0) - padding
                max_pt = np.max(pts_fg_clean, axis=0) + padding
                
                # === [新增] 应用 Z 轴偏移 ===
                # 正数表示整体往上抬，负数表示往下
                min_pt[2] += bbox_z_offset
                max_pt[2] += bbox_z_offset

                # 找出 BG 中位于该 Box 内的点
                if pts_bg.shape[0] > 0:
                    in_box_mask = np.all((pts_bg >= min_pt) & (pts_bg <= max_pt), axis=1)
                    
                    pts_bg_in_box = pts_bg[in_box_mask]
                    pts_bg_remain = pts_bg[~in_box_mask]
                    
                    pts_fg_final = np.vstack((pts_fg_clean, pts_bg_in_box))
                else:
                    pts_bg_remain = pts_bg
                    pts_fg_final = pts_fg_clean
            else:
                pts_fg_final = pts_fg_clean
                pts_bg_remain = pts_bg

            return pts_fg_final, pts_bg_remain
    
    def _process_point_cloud(self, sf, depth_m, fg_mask, w, h):
        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]
        
        valid = np.isfinite(depth_m) & (depth_m > self.depth_min) & (depth_m < self.depth_max)
        
        stride = max(1, self.pt_stride)
        us = np.arange(0, w, stride, dtype=np.int32)
        vs = np.arange(0, h, stride, dtype=np.int32)
        uu, vv = np.meshgrid(us, vs)
        
        sel = valid[vv, uu]
        if not np.any(sel):
            if self.publish_empty_clouds:
                # 如果是空云，尝试使用世界系的 Frame ID
                world_frame_id = sf.cam_pose.header.frame_id if sf.cam_pose.header.frame_id else "world"
                header_world = Header()
                header_world.stamp = sf.header.stamp
                header_world.frame_id = world_frame_id

                empty = self._empty_cloud(header_world)
                # 只在 en_vis=True 时发布独立的点云话题
                if self.en_vis:
                    self.pub_fg_cloud.publish(empty)
                    self.pub_bg_cloud.publish(empty)
                self._pub_sync_out(sf, empty, empty, fg_mask)
            self._republish_pose(sf)
            return

        is_fg = (fg_mask[vv, uu] > 0)
        fg_sel = is_fg & sel
        bg_sel = (~is_fg) & sel
        
        Z = depth_m[vv, uu]
        X = (uu - cx) * Z / fx
        Y = (vv - cy) * Z / fy
        
        def get_pts(selector):
            if not np.any(selector): return np.empty((0,3), dtype=np.float32)
            return np.stack([X[selector], Y[selector], Z[selector]], axis=1).astype(np.float32)

        # 1. 提取相机系下的点云
        pts_fg_cam = get_pts(fg_sel)
        pts_bg_cam = get_pts(bg_sel)
        
        # 2. 变换到世界系 (利用 cam_pose)
        pts_fg_world = self._transform_pts_to_world(pts_fg_cam, sf.cam_pose)
        pts_bg_world = self._transform_pts_to_world(pts_bg_cam, sf.cam_pose)

        # ==================== 调用改进后的函数 ====================
        # pts_fg_world, pts_bg_world = self._refine_fg_bg_clouds(
        #     pts_fg_world, 
        #     pts_bg_world, 
        #     nb_neighbors=20, 
        #     std_ratio=0.5, 
        #     padding=0.05,
        #     bbox_z_offset=0.0  # <--- 在这里设置 Z 轴偏移，例如往上抬 2cm
        # )
        # =========================================================

        # 3. 构建新的 Header (Frame ID 为世界系)
        world_frame_id = sf.cam_pose.header.frame_id if sf.cam_pose.header.frame_id else "world"
        header_world = Header()
        header_world.stamp = sf.header.stamp
        header_world.frame_id = world_frame_id

        # 4. 创建点云消息
        cloud_fg = self._create_cloud(header_world, pts_fg_world)
        cloud_bg = self._create_cloud(header_world, pts_bg_world)
        
        # 只在 en_vis=True 时发布独立的点云话题
        if self.en_vis:
            self.pub_fg_cloud.publish(cloud_fg)
            self.pub_bg_cloud.publish(cloud_bg)
        self._republish_pose(sf)
        
        # 将转换后的世界系点云和 mask 打包输出
        self._pub_sync_out(sf, cloud_fg, cloud_bg, fg_mask)

    def _create_cloud(self, header, pts):
        if pts.shape[0] == 0: return self._empty_cloud(header)
        fields = [PointField('x',0,7,1), PointField('y',4,7,1), PointField('z',8,7,1)]
        return pc2.create_cloud(header, fields, pts.tolist())

    def _empty_cloud(self, header):
        fields = [PointField('x',0,7,1), PointField('y',4,7,1), PointField('z',8,7,1)]
        return pc2.create_cloud(header, fields, [])

    def _pub_sync_out(self, sf, cfg, cbg, mask=None):
        if self.pub_sync_out.get_num_connections() > 0:
            out = SyncFrame()
            out.header = sf.header
            out.depth, out.rgb = sf.depth, sf.rgb
            out.tgt_pcl, out.env_pcl = cfg, cbg
            out.cam_pose = sf.cam_pose
            # 如果提供了 mask，将其包含在输出中
            if mask is not None:
                out.mask = numpy_to_image(mask, sf.header, "mono8")
            self.pub_sync_out.publish(out)

    def _republish_pose(self, sf):
        # 只在 en_vis=True 时发布 cam_pose
        if self.en_vis and self.pub_cam_pose.get_num_connections() > 0:
            odom = Odometry()
            odom.header.stamp = sf.header.stamp
            odom.header.frame_id = sf.header.frame_id
            odom.child_frame_id = sf.cam_pose.child_frame_id
            odom.pose = sf.cam_pose.pose
            odom.twist = sf.cam_pose.twist
            self.pub_cam_pose.publish(odom)

    def _publish_vis(self, header, bgr, mask):
        try:
            self.pub_mask.publish(numpy_to_image(mask, header, "mono8"))
            if self.pub_overlay.get_num_connections() > 0:
                overlay = bgr.copy()
                overlay[mask > 0, 2] = 255 # Red channel
                overlay = cv2.addWeighted(bgr, 1.0, overlay, self.alpha, 0)
                self.pub_overlay.publish(numpy_to_image(overlay, header, "bgr8"))
        except: pass

if __name__ == "__main__":
    rospy.init_node("yolo11_seg_pc", anonymous=False)
    node = YOLO11SegFromSyncFramePack()
    rospy.spin()