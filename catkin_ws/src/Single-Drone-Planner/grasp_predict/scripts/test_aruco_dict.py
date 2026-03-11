#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强诊断脚本：测试ArUco字典、图像预处理、以及检测普通棋盘格
"""

import rospy
import rosbag
import cv2
import numpy as np
import yaml
import os
from cv_bridge import CvBridge
import cv2.aruco as aruco

def test_dictionaries():
    # --- 1. 加载配置 ---
    cfg_path = "/home/jr/proj/airgrasp/planner_ws/src/Single-Drone-Planner/yolo_cup_seg/cfg/cfg_calibrate.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    bridge = CvBridge()
    debug_dir = "./hand_eye_debug"
    os.makedirs(debug_dir, exist_ok=True)
    
    # 所有可能的ArUco字典
    dict_names = [
        'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
        'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
        'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
        'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000',
        'DICT_ARUCO_ORIGINAL'
    ]
    
    bag = rosbag.Bag(cfg['bag_path'])
    print("成功打开 Bag: {}".format(cfg['bag_path']))
    
    # 读取一帧测试图像（使用第一个时间戳）
    test_stamp = cfg['stamps'][0]
    test_image = None
    
    for _, img_msg, t in bag.read_messages(
        topics=[cfg['topics']['image']],
        start_time=rospy.Time.from_sec(test_stamp - 0.1),
        end_time=rospy.Time.from_sec(test_stamp + 0.1)
    ):
        test_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        break
    
    bag.close()
    
    if test_image is None:
        print("错误: 无法读取测试图像")
        return
    
    # 保存原始图像
    cv2.imwrite(os.path.join(debug_dir, "test_original.jpg"), test_image)
    print("原始图像已保存: {}/test_original.jpg".format(debug_dir))
    print("图像尺寸: {}x{}\n".format(test_image.shape[1], test_image.shape[0]))
    
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    
    # === 1. 测试标准ArUco检测 ===
    print("="*60)
    print("【方法1】标准ArUco检测（所有字典）")
    print("="*60)
    
    results = []
    for dict_name in dict_names:
        try:
            dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
            corners, ids, _ = aruco.detectMarkers(gray, dictionary)
            
            if ids is not None and len(ids) > 0:
                result = {
                    'dict_name': dict_name,
                    'num_markers': len(ids),
                    'marker_ids': sorted(ids.flatten().tolist())
                }
                results.append(result)
                print("[✓] {}: 检测到 {} 个标记".format(dict_name, len(ids)))
        except Exception as e:
            pass
    
    if len(results) == 0:
        print("[✗] 标准方法未检测到任何ArUco标记\n")
    
    # === 2. 尝试图像增强后检测 ===
    print("="*60)
    print("【方法2】图像预处理增强后检测")
    print("="*60)
    
    # 2a. 自适应阈值
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(os.path.join(debug_dir, "test_adaptive.jpg"), adaptive)
    
    # 2b. 锐化
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    cv2.imwrite(os.path.join(debug_dir, "test_sharpened.jpg"), sharpened)
    
    # 2c. 对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    cv2.imwrite(os.path.join(debug_dir, "test_enhanced.jpg"), enhanced)
    
    preprocessed_images = [
        ("自适应阈值", adaptive),
        ("锐化", sharpened),
        ("对比度增强", enhanced)
    ]
    
    enhanced_results = []
    for preprocess_name, img in preprocessed_images:
        for dict_name in ['DICT_6X6_250', 'DICT_4X4_50', 'DICT_5X5_100']:  # 只测试常用的
            try:
                dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
                corners, ids, _ = aruco.detectMarkers(img, dictionary)
                
                if ids is not None and len(ids) > 0:
                    print("[✓] {} + {}: 检测到 {} 个标记".format(
                        preprocess_name, dict_name, len(ids)))
                    enhanced_results.append((preprocess_name, dict_name, len(ids)))
            except:
                pass
    
    if len(enhanced_results) == 0:
        print("[✗] 预处理后仍未检测到ArUco标记\n")
    
    # === 3. 检测是否为普通棋盘格 ===
    print("="*60)
    print("【方法3】检测普通棋盘格标定板")
    print("="*60)
    
    chessboard_sizes = [(9,6), (8,6), (7,6), (6,6), (9,7), (8,8), (7,5)]
    chessboard_found = False
    
    for size in chessboard_sizes:
        ret, corners = cv2.findChessboardCorners(gray, size, None)
        if ret:
            print("[✓] 检测到棋盘格: {}x{} ({} 个角点)".format(
                size[0], size[1], len(corners)))
            chessboard_found = True
            
            # 绘制并保存
            vis = test_image.copy()
            cv2.drawChessboardCorners(vis, size, corners, ret)
            cv2.imwrite(os.path.join(debug_dir, "chessboard_detected.jpg"), vis)
            print("    已保存到: {}/chessboard_detected.jpg".format(debug_dir))
    
    if not chessboard_found:
        print("[✗] 也未检测到普通棋盘格\n")
    
    # === 总结 ===
    print("\n" + "="*60)
    print("诊断总结")
    print("="*60)
    
    if len(results) > 0:
        best = max(results, key=lambda x: x['num_markers'])
        print("\n✓ 找到可用的ArUco字典: {}".format(best['dict_name']))
        print("  检测到 {} 个标记".format(best['num_markers']))
        print("\n请修改配置文件:")
        print("  charuco:")
        print("    dictionary: \"{}\"".format(best['dict_name']))
    elif len(enhanced_results) > 0:
        print("\n✓ 使用图像预处理后可以检测到标记")
        print("  需要在代码中添加预处理步骤")
    elif chessboard_found:
        print("\n⚠ 这是一个普通的棋盘格标定板，不是ChArUco板！")
        print("  你需要使用 cv2.calibrateCamera() 而不是 ChArUco 标定")
        print("  或者更换为 ChArUco 标定板")
    else:
        print("\n✗ 完全无法检测到标定板")
        print("\n可能的原因:")
        print("  1. 图像分辨率太低或压缩严重")
        print("  2. 标定板在图像中太小或角度太斜")
        print("  3. 光照或对比度问题")
        print("  4. 使用了自定义标定板")
        print("\n建议:")
        print("  - 查看保存的图像: {}/test_*.jpg".format(debug_dir))
        print("  - 尝试更近距离拍摄标定板")
        print("  - 确保标定板清晰可见且占据较大视野")

if __name__ == "__main__":
    test_dictionaries()
