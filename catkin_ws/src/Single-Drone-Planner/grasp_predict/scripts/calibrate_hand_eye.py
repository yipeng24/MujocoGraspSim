#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import rosbag
import cv2
import numpy as np
import yaml
import os
from cv_bridge import CvBridge
import cv2.aruco as aruco
from tf.transformations import quaternion_matrix

def solve_hand_eye():
    # --- 1. 加载配置 ---
    cfg_path = "/home/jr/proj/airgrasp/planner_ws/src/Single-Drone-Planner/yolo_cup_seg/cfg/cfg_calibrate.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    bridge = CvBridge()
    
    # 创建调试输出目录
    debug_dir = "./hand_eye_debug"
    os.makedirs(debug_dir, exist_ok=True)
    print("调试图像将保存到: {}".format(debug_dir))
    
    # 初始化 ChArUco 标定板
    c_cfg = cfg['charuco']
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, c_cfg['dictionary']))
    board = aruco.CharucoBoard_create(
        c_cfg['squares_x'], c_cfg['squares_y'], 
        c_cfg['square_length'], c_cfg['marker_length'], 
        dictionary
    )

    bag = rosbag.Bag(cfg['bag_path'])
    print("成功打开 Bag: {}".format(cfg['bag_path']))

    # --- 2. 从配置文件读取相机内参 ---
    cam_matrix = np.array(cfg['camera']['K']).reshape(3, 3)
    dist_coeffs = np.array(cfg['camera']['D']) if cfg['camera']['D'] else np.zeros(5)
    print("从配置文件读取相机内参: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}".format(
        cam_matrix[0, 0], cam_matrix[1, 1], cam_matrix[0, 2], cam_matrix[1, 2]))

    # --- 3. 预加载所有位姿数据 ---
    print("预加载位姿数据...")
    all_poses = []
    for _, msg, t in bag.read_messages(topics=[cfg['topics']['pose']]):
        all_poses.append({'t': t.to_sec(), 'msg': msg.transform})

    # 存储用于标定的数据
    R_gripper2base, t_gripper2base = [], []
    R_target2cam, t_target2cam = [], []

    # --- 4. 遍历时间戳提取数据 ---
    print("开始根据时间戳提取图像并检测 ChArUco...")
    sample_idx = 0
    for stamp in cfg['stamps']:
        found_sample = False
        # 在指定时间戳前后 0.1s 搜索
        for _, img_msg, t in bag.read_messages(
            topics=[cfg['topics']['image']],
            start_time=rospy.Time.from_sec(stamp - 0.1),
            end_time=rospy.Time.from_sec(stamp + 0.1)
        ):
            cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            debug_img = cv_image.copy()
            
            # 检测 ArUco 码
            corners, ids, _ = aruco.detectMarkers(gray, dictionary)
            
            # 绘制检测到的ArUco标记用于调试
            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(debug_img, corners, ids)
                
                # 核心：插值获取 ChArUco 角点 (这是提高精度的关键)
                ret, c_corners, c_ids = aruco.interpolateCornersCharuco(
                    corners, ids, gray, board, cameraMatrix=cam_matrix, distCoeffs=dist_coeffs
                )
                
                print("  Stamp {}: 检测到 {} 个ArUco标记, {} 个ChArUco角点".format(
                    stamp, len(ids), ret if ret else 0))
                
                if ret and ret > 4: # 至少需要4个点来估计位姿
                    # 绘制ChArUco角点
                    aruco.drawDetectedCornersCharuco(debug_img, c_corners, c_ids)
                    
                    # 估计标定板位姿
                    retval, rvec, tvec = aruco.estimatePoseCharucoBoard(
                        c_corners, c_ids, board, cam_matrix, dist_coeffs, None, None
                    )
                    
                    if retval:
                        # 绘制坐标轴
                        aruco.drawAxis(debug_img, cam_matrix, dist_coeffs, rvec, tvec, 0.1)
                        
                        # 寻找最匹配的机器人位姿
                        closest_p = min(all_poses, key=lambda x: abs(x['t'] - stamp))
                        
                        # 转换机器人位姿 (EE to Base)
                        q = [closest_p['msg'].rotation.x, closest_p['msg'].rotation.y, 
                             closest_p['msg'].rotation.z, closest_p['msg'].rotation.w]
                        tr = closest_p['msg'].translation
                        T_b_e = quaternion_matrix(q)
                        
                        # 保存 A (Robot) 和 B (Camera)
                        R_gripper2base.append(T_b_e[:3, :3])
                        t_gripper2base.append(np.array([tr.x, tr.y, tr.z]).reshape(3, 1))
                        R_target2cam.append(cv2.Rodrigues(rvec)[0])
                        t_target2cam.append(tvec)
                        
                        # 保存成功检测的图像
                        success_path = os.path.join(debug_dir, "success_{:02d}.jpg".format(sample_idx))
                        cv2.imwrite(success_path, debug_img)
                        print("  [成功] 检测到 {} 个角点，已保存到 {}".format(ret, success_path))
                        found_sample = True
                        sample_idx += 1
                        break
                    else:
                        print("  [警告] ChArUco角点足够但位姿估计失败")
                else:
                    print("  [失败] ChArUco角点不足 (需要>4个)")
            else:
                print("  Stamp {}: 未检测到任何ArUco标记".format(stamp))
            
            # 保存调试图像（即使失败也保存）
            if not found_sample:
                fail_path = os.path.join(debug_dir, "fail_{:02d}.jpg".format(sample_idx))
                cv2.imwrite(fail_path, debug_img)
                print("  调试图像已保存到: {}".format(fail_path))
                sample_idx += 1
            break
        
        if not found_sample:
            print("  [失败] Stamp {}: 无法检测到足够的特征点\n".format(stamp))

    bag.close()

    # --- 5. 执行手眼标定求解 ---
    if len(R_gripper2base) < 5:
        print("\n有效样本不足 (仅 {})，无法进行标定。".format(len(R_gripper2base)))
        return

    print("\n运行标定算法 (TSAI)...")
    R_ee_cam, t_ee_cam = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # --- 6. 结果输出 ---
    T_ee_cam = np.eye(4)
    T_ee_cam[:3, :3] = R_ee_cam
    T_ee_cam[:3, 3] = t_ee_cam.flatten()

    print("\n" + "="*50)
    print("标定结果: 相机相对于机械臂末端的外参 (T_ee_cam)")
    print("="*50)
    print(np.array2string(T_ee_cam, separator=', '))
    print("\n平移向量 (x, y, z) 单位: 米")
    print(t_ee_cam.flatten())
    print("="*50)

if __name__ == "__main__":
    solve_hand_eye()