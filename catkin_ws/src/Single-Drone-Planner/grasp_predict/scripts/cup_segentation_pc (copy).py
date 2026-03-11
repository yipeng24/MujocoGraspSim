#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from quadrotor_msgs.msg import SyncFrame


def parse_K_param(default_K):
    K_param = rospy.get_param("~K", default_K)
    if isinstance(K_param, str):
        K_list = [float(x) for x in K_param.replace("[","").replace("]","").replace(","," ").split()]
    elif isinstance(K_param, (list, tuple)) and len(K_param) == 9:
        K_list = [float(x) for x in K_param]
    else:
        K_list = default_K
    K = np.array(K_list, dtype=np.float64).reshape(3, 3)
    return K


class YOLO11SegFromSyncFramePack:
    def __init__(self):
        # -------- 参数 --------
        self.sync_frame_topic = rospy.get_param("~sync_frame_topic", "/sync_frame")
        self.model_path = rospy.get_param("~model_path", "$(find yolo11_cup_seg)/weights/best.engine")
        self.conf   = float(rospy.get_param("~conf", 0.25))
        self.imgsz  = int(rospy.get_param("~imgsz", 640))
        self.device = rospy.get_param("~device", "auto")   # "auto"|"cuda"|"cpu"
        self.alpha  = float(rospy.get_param("~alpha", 0.5))

        # 深度&点云
        self.depth_encoding = rospy.get_param("~depth_encoding", "auto")  # "auto"|"16UC1"|"32FC1"
        self.depth_scale    = float(rospy.get_param("~depth_scale", 0.001))
        self.depth_min      = float(rospy.get_param("~depth_min", 0.1))
        self.depth_max      = float(rospy.get_param("~depth_max", 10.0))
        self.pt_stride      = int(rospy.get_param("~point_stride", 2))
        self.publish_empty_clouds = bool(rospy.get_param("~publish_empty_clouds", True))

        # 类别过滤（默认：全部实例=前景）
        fg_class_ids_param = rospy.get_param("~fg_class_ids", "auto")
        if fg_class_ids_param in ("auto", "", "none", "None"):
            self.fg_class_ids = None
        else:
            try:
                self.fg_class_ids = [int(x) for x in str(fg_class_ids_param).split(",")]
            except Exception:
                self.fg_class_ids = None

        # 固定相机内参（不订阅 CameraInfo）
        default_K = [607.2257690429688, 0.0, 315.67559814453125,
                     0.0, 607.5105590820312, 233.73883056640625,
                     0.0, 0.0, 1.0]
        self.K = parse_K_param(default_K)

        # -------- 模型 --------
        rospy.loginfo(f"[PackNode] Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model.to(self.device)
        except Exception as e:
            rospy.logwarn(f"[PackNode] model.to({self.device}) failed/skipped: {e}")
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        self.bridge = CvBridge()

        # -------- 发布者（单独话题 + 聚合 /sync_frame_out）--------
        self.pub_mask      = rospy.Publisher("~mask", Image, queue_size=1)
        self.pub_overlay   = rospy.Publisher("~overlay", Image, queue_size=1)
        self.pub_fg_cloud  = rospy.Publisher("/sync_tgt_pcl", PointCloud2, queue_size=1)
        self.pub_bg_cloud  = rospy.Publisher("/sync_env_pcl", PointCloud2, queue_size=1)
        self.pub_cam_pose  = rospy.Publisher("/sync_cam_pose", Odometry, queue_size=1)

        self.pub_sync_out  = rospy.Publisher("/sync_frame_out", SyncFrame, queue_size=1)

        # -------- 订阅自定义 SyncFrame --------
        self.sub_sync = rospy.Subscriber(self.sync_frame_topic, SyncFrame, self.cb_syncframe,
                                         queue_size=1, buff_size=2**24)

        fx, fy, cx, cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]
        rospy.loginfo(f"[PackNode] Using fixed K: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
        rospy.loginfo(f"[PackNode] device={self.device}, imgsz={self.imgsz}, conf={self.conf}, stride={self.pt_stride}")

    # ---------- 主回调：SyncFrame（输入RGB/Depth/pose，输出点云+聚合SyncFrame） ----------
    def cb_syncframe(self, sf: SyncFrame):
        # 1) 解码 RGB
        try:
            bgr = self.bridge.imgmsg_to_cv2(sf.sync_rgb, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"[PackNode] cv_bridge rgb failed: {e}")
            return

        # 2) 解码深度（米）
        try:
            enc = sf.sync_depth.encoding if self.depth_encoding == "auto" else self.depth_encoding
            if enc == "16UC1" or ("16UC1" in sf.sync_depth.encoding and self.depth_encoding == "auto"):
                depth_raw = self.bridge.imgmsg_to_cv2(sf.sync_depth, desired_encoding="passthrough")
                depth_m = depth_raw.astype(np.float32) * self.depth_scale
            else:
                depth_m = self.bridge.imgmsg_to_cv2(sf.sync_depth, desired_encoding="32FC1")
        except Exception as e:
            rospy.logwarn(f"[PackNode] cv_bridge depth failed: {e}")
            return

        h, w = bgr.shape[:2]
        if depth_m.shape[0] != h or depth_m.shape[1] != w:
            depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

        # 3) YOLOv11-seg 推理
        try:
            res = self.model.predict(source=bgr, imgsz=self.imgsz, conf=self.conf,
                                     device=self.device, verbose=False)[0]
        except Exception as e:
            rospy.logwarn(f"[PackNode] YOLO predict failed: {e}")
            return

        # 4) 构建二值前景 mask（掩膜->原图尺寸）
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        try:
            masks = getattr(res, "masks", None)
            boxes = getattr(res, "boxes", None)
            n_classes = len(self.model.names) if hasattr(self.model, "names") else 1
            fg_ids = self._auto_fg_ids(n_classes)

            if masks is not None and masks.data is not None and masks.data.shape[0] > 0:
                mdata = masks.data
                cls_ids = None
                if boxes is not None and boxes.cls is not None:
                    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)

                for i in range(mdata.shape[0]):
                    take = True
                    if cls_ids is not None and fg_ids is not None:
                        take = int(cls_ids[i]) in fg_ids
                    if not take:
                        continue
                    m = mdata[i].detach().cpu().numpy()
                    if m.dtype != np.bool_:
                        m = (m > 0.5).astype(np.uint8)
                    if m.shape[0] != h or m.shape[1] != w:
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    fg_mask[m > 0] = 255
        except Exception as e:
            rospy.logwarn(f"[PackNode] build mask failed: {e}")

        # 5) 可视化（有订阅者才生成）
        self._publish_mask_and_overlay(sf.header, bgr, fg_mask)

        # 6) 点云生成（向量化）
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        valid = np.isfinite(depth_m) & (depth_m > self.depth_min) & (depth_m < self.depth_max)

        stride = max(1, self.pt_stride)
        us = np.arange(0, w, stride, dtype=np.int32)
        vs = np.arange(0, h, stride, dtype=np.int32)
        uu, vv = np.meshgrid(us, vs)

        sel = valid[vv, uu]
        if not np.any(sel):
            if self.publish_empty_clouds:
                empty = self._empty_cloud(sf.header, sf.header.frame_id)
                self.pub_fg_cloud.publish(empty)
                self.pub_bg_cloud.publish(empty)
                self._publish_syncframe_out(sf, empty, empty)
            self._republish_pose(sf)
            return

        fg_sel = (fg_mask[vv, uu] > 0) & sel
        bg_sel = (~(fg_mask[vv, uu] > 0)) & sel

        Z = depth_m[vv, uu]
        X = (uu.astype(np.float32) - cx) / fx * Z
        Y = (vv.astype(np.float32) - cy) / fy * Z

        pts_fg = np.stack([X[fg_sel], Y[fg_sel], Z[fg_sel]], axis=1).astype(np.float32) if np.any(fg_sel) else np.empty((0, 3), dtype=np.float32)
        pts_bg = np.stack([X[bg_sel], Y[bg_sel], Z[bg_sel]], axis=1).astype(np.float32) if np.any(bg_sel) else np.empty((0, 3), dtype=np.float32)

        frame_id = sf.header.frame_id or "camera_color_optical_frame"
        stamp = sf.header.stamp

        cloud_fg = self._empty_cloud(sf.header, frame_id) if pts_fg.shape[0] == 0 else self._to_cloud_msg(pts_fg, frame_id, stamp)
        cloud_bg = self._empty_cloud(sf.header, frame_id) if pts_bg.shape[0] == 0 else self._to_cloud_msg(pts_bg, frame_id, stamp)

        # 单独话题发布
        self.pub_fg_cloud.publish(cloud_fg)
        self.pub_bg_cloud.publish(cloud_bg)
        self._republish_pose(sf)

        # 7) 组装并发布新的 SyncFrame（一次性带出两路点云 + 位姿）
        self._publish_syncframe_out(sf, cloud_fg, cloud_bg)

    # ---------- 聚合 SyncFrame 输出 ----------
    def _publish_syncframe_out(self, sf_in: SyncFrame, cloud_fg: PointCloud2, cloud_bg: PointCloud2):
        if self.pub_sync_out.get_num_connections() == 0:
            return
        out = SyncFrame()
        out.header = sf_in.header            # 时间戳 / frame_id 与输入一致
        out.sync_depth = sf_in.sync_depth    # 透传输入的图像
        out.sync_rgb = sf_in.sync_rgb
        out.sync_tgt_pcl = cloud_fg          # 我们刚刚生成的点云
        out.sync_env_pcl = cloud_bg
        out.sync_cam_pose = sf_in.sync_cam_pose
        self.pub_sync_out.publish(out)

    # ---------- Pose 透传 ----------
    def _republish_pose(self, sf: SyncFrame):
        if self.pub_cam_pose.get_num_connections() == 0:
            return
        odom = Odometry()
        odom.header = Header()
        odom.header.stamp = sf.header.stamp
        odom.header.frame_id = sf.header.frame_id
        odom.child_frame_id = sf.sync_cam_pose.child_frame_id
        odom.pose = sf.sync_cam_pose.pose
        odom.twist = sf.sync_cam_pose.twist
        self.pub_cam_pose.publish(odom)
        print("pub")

    # ---------- 工具 ----------
    def _auto_fg_ids(self, n_classes: int):
        if self.fg_class_ids is not None:
            return self.fg_class_ids
        if n_classes <= 1:
            return [0]  # 单类分割：唯一类=前景
        return None    # 多类：全部实例=前景

    def _publish_mask_and_overlay(self, header, bgr, fg_mask):
        # mask
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(fg_mask, encoding="mono8")
            mask_msg.header = header
            self.pub_mask.publish(mask_msg)
        except Exception as e:
            rospy.logwarn(f"[PackNode] publish mask failed: {e}")

        # overlay（仅在有订阅者时生成）
        if self.pub_overlay.get_num_connections() > 0:
            try:
                color = np.zeros_like(bgr, dtype=np.uint8)
                color[:, :, 2] = 255
                mask_3 = cv2.merge([fg_mask, fg_mask, fg_mask])
                colored = np.where(mask_3 > 0, color, 0)
                overlay = cv2.addWeighted(bgr, 1.0, colored, self.alpha, 0)
                overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
                overlay_msg.header = header
                self.pub_overlay.publish(overlay_msg)
            except Exception as e:
                rospy.logwarn(f"[PackNode] publish overlay failed: {e}")

    def _to_cloud_msg(self, pts_xyz: np.ndarray, frame_id: str, stamp):
        fields = [
            PointField('x', 0,  PointField.FLOAT32, 1),
            PointField('y', 4,  PointField.FLOAT32, 1),
            PointField('z', 8,  PointField.FLOAT32, 1),
        ]
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return pc2.create_cloud(header, fields, pts_xyz.tolist())

    def _empty_cloud(self, header_in, frame_id=None):
        fields = [
            PointField('x', 0,  PointField.FLOAT32, 1),
            PointField('y', 4,  PointField.FLOAT32, 1),
            PointField('z', 8,  PointField.FLOAT32, 1),
        ]
        header = Header()
        header.stamp = header_in.stamp
        header.frame_id = frame_id or (header_in.frame_id if header_in.frame_id else "camera_color_optical_frame")
        return pc2.create_cloud(header, fields, [])


def main():
    rospy.init_node("yolo11_seg_from_syncframe_pack", anonymous=False)
    YOLO11SegFromSyncFramePack()
    rospy.loginfo("[PackNode] Spinning ...")
    rospy.spin()


if __name__ == "__main__":
    main()

