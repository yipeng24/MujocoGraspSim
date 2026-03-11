#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import rosbag
import cv2
import numpy as np
import yaml
import os
from cv_bridge import CvBridge
from tf.transformations import quaternion_matrix

try:
    from pupil_apriltags import Detector
except ImportError:
    print("错误: 未安装 pupil-apriltags 库 (pip install pupil-apriltags)")
    import sys
    sys.exit(1)


class AprilGridCalibrator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bridge = CvBridge()

        # AprilGrid 配置
        grid_cfg = cfg['aprilgrid']
        self.cols = int(grid_cfg['cols'])
        self.rows = int(grid_cfg['rows'])
        self.tag_size = float(grid_cfg['tag_size'])
        self.tag_spacing = float(grid_cfg['tag_spacing'])

        self.cfg_families = grid_cfg.get('families', None)
        self.cfg_id_offset = grid_cfg.get('id_offset', None)

        # 相机内参
        self.cam_matrix = np.array(cfg['camera']['K'], dtype=np.float64).reshape(3, 3)
        D = cfg['camera'].get('D', None)
        if D is None or len(D) == 0:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        else:
            self.dist_coeffs = np.array(D, dtype=np.float64).reshape(-1, 1)

        # AprilGrid 3D 点
        self.object_points = self._generate_grid_points()

        # 常用 family 候选
        self.family_candidates = []
        if self.cfg_families:
            for f in str(self.cfg_families).split(","):
                f = f.strip()
                if f:
                    self.family_candidates.append(f)

        if not self.family_candidates:
            # self.family_candidates = [
            #     "tag36h11", "tag25h9", "tag16h5",
            #     "tagStandard41h12", "tagCircle21h7"
            # ]
            self.family_candidates = [
                "tag36h11"
            ]


        self.detector_params = dict(
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        self.detectors = {}

    def _get_detector(self, family):
        if family not in self.detectors:
            self.detectors[family] = Detector(families=family, **self.detector_params)
        return self.detectors[family]

    def _generate_grid_points(self):
        points = []
        spacing_m = self.tag_size * (1.0 + self.tag_spacing)
        half = self.tag_size / 2.0
        for r in range(self.rows):
            for c in range(self.cols):
                cx = c * spacing_m
                cy = r * spacing_m
                corners = [
                    [cx - half, cy - half, 0.0],
                    [cx + half, cy - half, 0.0],
                    [cx + half, cy + half, 0.0],
                    [cx - half, cy + half, 0.0],
                ]
                points.extend(corners)
        return np.array(points, dtype=np.float32)

    def _enhance(self, gray):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _match_points_with_offset(self, detections, id_offset):
        # 预检查：如果同一个 ID 出现多次，说明有误检，直接舍弃这组检测
        ids = [d.tag_id for d in detections]
        if len(ids) != len(set(ids)):
            return None, None, []

        det_dict = {d.tag_id: d for d in detections}
        img_pts = []
        obj_pts = []
        used_ids = []

        n = self.rows * self.cols
        for i in range(n):
            tid = i + id_offset
            if tid in det_dict:
                d = det_dict[tid]
                img_pts.extend(d.corners.tolist())
                obj_pts.extend(self.object_points[i*4:(i+1)*4].tolist())
                used_ids.append(tid)

        if len(img_pts) == 0:
            return None, None, []

        return (np.array(obj_pts, dtype=np.float32),
                np.array(img_pts, dtype=np.float32),
                used_ids)

    def detect_aprilgrid(self, image, idx, debug_dir):
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"raw_{idx:02d}.jpg"), image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_clahe = self._enhance(gray)

        best = {
            "family": None, "mode": None, "detections": [],
            "obj_pts": None, "img_pts": None, "used_ids": [], "score": -1
        }

        for fam in self.family_candidates:
            det = self._get_detector(fam)
            for mode_name, g in [("raw", gray), ("clahe", gray_clahe)]:
                ds = det.detect(g)
                if len(ds) == 0: continue

                # 自动尝试 offset
                found_ids = sorted([d.tag_id for d in ds])
                min_id = min(found_ids)
                offsets = [0, int(min_id)]
                if self.cfg_id_offset is not None:
                    offsets.insert(0, int(self.cfg_id_offset))
                offsets = list(dict.fromkeys(offsets))

                for off in offsets:
                    obj_pts, img_pts, used_ids = self._match_points_with_offset(ds, off)
                    if obj_pts is None: continue
                    score = len(used_ids)
                    if score > best["score"]:
                        best.update({
                            "family": fam, "mode": mode_name, "detections": ds,
                            "obj_pts": obj_pts, "img_pts": img_pts,
                            "used_ids": used_ids, "score": score
                        })

        if best["score"] <= 0 or best["obj_pts"] is None:
            print(f"    [Index {idx}] 未检测到有效 Tag 布局")
            return False, None, image

        print(f"    [Index {idx}] family={best['family']} mode={best['mode']} "
              f"匹配到 {best['score']} 个 tags, IDs={sorted(best['used_ids'])}")

        vis = image.copy()
        for d in best["detections"]:
            pts = d.corners.astype(int)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cv2.putText(vis, str(d.tag_id), tuple(pts[0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return True, (best["obj_pts"], best["img_pts"]), vis

    def calibrate(self, bag_path, timestamps, pose_topic, image_topic, debug_dir="./hand_eye_debug"):
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        bag = rosbag.Bag(bag_path)
        print("正在从 Bag 提取位姿数据...")
        all_poses = []
        for _, msg, t in bag.read_messages(topics=[pose_topic]):
            all_poses.append({'t': t.to_sec(), 'msg': msg})

        if len(all_poses) == 0:
            print("错误：pose_topic 没读到消息")
            bag.close()
            return

        R_gripper2base, t_gripper2base = [], []
        R_target2cam, t_target2cam = [], []

        print(f"开始处理 {len(timestamps)} 个时间戳...")
        for i, stamp in enumerate(timestamps):
            success_at_this_stamp = False

            for _, img_msg, t in bag.read_messages(
                topics=[image_topic],
                start_time=rospy.Time.from_sec(stamp - 0.5),
                end_time=rospy.Time.from_sec(stamp + 0.5)
            ):
                cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
                success, points, vis_img = self.detect_aprilgrid(cv_img, i, debug_dir)
                if not success: continue

                obj_pts, img_pts = points
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, self.cam_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok: continue

                # --- 修复核心：调用 extract_rot_trans 提取位姿 ---
                closest_p = min(all_poses, key=lambda x: abs(x['t'] - t.to_sec()))
                try:
                    quat, trans = self.extract_rot_trans(closest_p['msg'])
                    T_b_e = quaternion_matrix(quat)
                    
                    R_gripper2base.append(T_b_e[:3, :3])
                    t_gripper2base.append(np.array(trans).reshape(3, 1))

                    R_tc, _ = cv2.Rodrigues(rvec)
                    R_target2cam.append(R_tc.copy())
                    t_target2cam.append(np.array(tvec, dtype=np.float64).reshape(3, 1))

                    cv2.drawFrameAxes(vis_img, self.cam_matrix, self.dist_coeffs, rvec, tvec, 0.1)
                    cv2.imwrite(os.path.join(debug_dir, f"result_{i:02d}.jpg"), vis_img)
                    print(f"[成功] 索引 {i}")
                    success_at_this_stamp = True
                    break
                except Exception as e:
                    print(f"[错误] 提取位姿失败: {e}")
                    break

            if not success_at_this_stamp:
                print(f"[失败] 索引 {i}")

        bag.close()

        if len(R_gripper2base) < 3:
            print(f"\n有效数据不足 ({len(R_gripper2base)})，无法进行手眼标定。")
            return

        R_ee_cam, t_ee_cam = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        T_ee_cam = np.eye(4, dtype=np.float64)
        T_ee_cam[:3, :3] = R_ee_cam
        T_ee_cam[:3, 3] = t_ee_cam.flatten()

        print("\n" + "=" * 60)
        print("标定成功! T_end_effector_to_camera (即相机在手眼末端坐标系下的位姿):")
        print(T_ee_cam)
        print("=" * 60)

    @staticmethod
    def extract_rot_trans(msg):
        """
        兼容多种 ROS 位姿消息
        """
        # TransformStamped
        if hasattr(msg, "transform"):
            r = msg.transform.rotation
            t = msg.transform.translation
            return (r.x, r.y, r.z, r.w), (t.x, t.y, t.z)
        # Transform
        if hasattr(msg, "rotation") and hasattr(msg, "translation"):
            r = msg.rotation
            t = msg.translation
            return (r.x, r.y, r.z, r.w), (t.x, t.y, t.z)
        # PoseStamped
        if hasattr(msg, "pose"):
            o = msg.pose.orientation
            p = msg.pose.position
            return (o.x, o.y, o.z, o.w), (p.x, p.y, p.z)
        # Pose
        if hasattr(msg, "orientation") and hasattr(msg, "position"):
            o = msg.orientation
            p = msg.position
            return (o.x, o.y, o.z, o.w), (p.x, p.y, p.z)
        raise TypeError(f"不支持的消息类型: {type(msg)}")


def main():
    cfg_path = "/home/jr/proj/airgrasp/planner_ws/src/Single-Drone-Planner/yolo_cup_seg/cfg/cfg_calibrate.yaml"
    if not os.path.exists(cfg_path):
        print(f"找不到配置文件: {cfg_path}")
        return

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    calibrator = AprilGridCalibrator(cfg)
    calibrator.calibrate(
        cfg['bag_path'],
        cfg['stamps'],
        cfg['topics']['pose'],
        cfg['topics']['image'],
        debug_dir=cfg.get('debug_dir', "./hand_eye_debug")
    )


if __name__ == "__main__":
    main()