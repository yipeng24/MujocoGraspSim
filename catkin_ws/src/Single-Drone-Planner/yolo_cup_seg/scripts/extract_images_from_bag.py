#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import rosbag
from ultralytics import YOLO
import torch
import sys

# -----------------------------------------------------------
# 1. 模拟或替换 ros_numpy 核心解码功能
# -----------------------------------------------------------
def fast_numpify_depth(msg):
    """处理 16UC1 深度图像消息"""
    depth_data = np.frombuffer(msg.data, dtype=np.uint16)
    return depth_data.reshape(msg.height, msg.width)

def fast_numpify_rgb(msg):
    """处理 Image 原始图像消息 (bgr8/rgb8)"""
    img_data = np.frombuffer(msg.data, dtype=np.uint8)
    img = img_data.reshape(msg.height, msg.width, 3)
    if "rgb8" in msg.encoding:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# -----------------------------------------------------------
# 2. 加载配置管理器
# -----------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from data_manager import ConfigManager

class BagProcessor:
    def __init__(self):
        # 初始化配置管理器
        parent_dir = os.path.dirname(current_dir)
        config_file = os.path.join(parent_dir, "cfg", "config.yaml")
        
        if not os.path.exists(config_file):
            print(f"[ERROR] 配置文件不存在: {config_file}")
            sys.exit(1)
            
        self.cfg = ConfigManager(config_file)

        # 从 Config 获取所有参数
        self.bag_path = self.cfg.get("bag_settings.path", "")
        self.output_base = self.cfg.get("bag_settings.output_dir", "./output")
        self.rgb_topic = self.cfg.get("topics.rgb", "/camera/image/compressed")
        self.depth_topic = self.cfg.get("topics.depth", "/camera/depth/image_raw")
        self.time_threshold = float(self.cfg.get("sync.time_threshold", 0.05))

        # 模型参数
        self.model_path = self.cfg.get("model.path", "")
        self.conf       = float(self.cfg.get("model.conf", 0.5))
        self.imgsz      = int(self.cfg.get("model.imgsz", 640))
        self.device     = self.cfg.get("model.device", "auto")
        
        fg_ids_param = self.cfg.get("point_cloud.fg_class_ids", "auto")
        self.fg_class_ids = self._parse_fg_ids(fg_ids_param)

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[INFO] 正在加载 YOLO 模型: {self.model_path}")
        self.model = YOLO(self.model_path).to(self.device)

    def _parse_fg_ids(self, param):
        if str(param).lower() in ("auto", "", "none"): return None
        try:
            if isinstance(param, list): return [int(x) for x in param]
            return [int(x) for x in str(param).split(",")]
        except: return None

    def run(self):
        if not os.path.exists(self.bag_path):
            print(f"[ERROR] Bag 文件不存在: {self.bag_path}")
            return

        dirs = {
            "rgb": os.path.join(self.output_base, "rgb"),
            "depth": os.path.join(self.output_base, "depth"),
            "mask": os.path.join(self.output_base, "mask")
        }
        for d in dirs.values():
            if not os.path.exists(d): os.makedirs(d)

        bag = rosbag.Bag(self.bag_path, 'r')
        
        # 索引深度图
        depth_msgs = []
        for _, msg, _ in bag.read_messages(topics=[self.depth_topic]):
            depth_msgs.append((msg.header.stamp.to_sec(), msg))
        
        if not depth_msgs:
            print("[ERROR] 未在 Bag 中找到深度图数据!")
            bag.close()
            return
        
        depth_stamps = np.array([m[0] for m in depth_msgs])
        saved_count = 0

        print(f"[INFO] 开始处理并提取三通道 Mask...")
        for _, msg, _ in bag.read_messages(topics=[self.rgb_topic]):
            rgb_t = msg.header.stamp.to_sec()
            idx = np.argmin(np.abs(depth_stamps - rgb_t))
            diff = abs(depth_stamps[idx] - rgb_t)

            if diff < self.time_threshold:
                try:
                    # 解码图像
                    if "CompressedImage" in msg._type:
                        cv_rgb = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                    else:
                        cv_rgb = fast_numpify_rgb(msg)

                    cv_depth = fast_numpify_depth(depth_msgs[idx][1])

                    # 推理
                    res = self.model.predict(cv_rgb, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
                    
                    # --- [修改部分] 生成三通道抠图 Mask ---
                    h, w = cv_rgb.shape[:2]
                    # 初始化二值掩码（单通道）
                    binary_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    if res.masks is not None:
                        clss = res.boxes.cls.cpu().numpy().astype(int)
                        for i, m_t in enumerate(res.masks.data):
                            if self.fg_class_ids and clss[i] not in self.fg_class_ids: 
                                continue
                            m_np = cv2.resize(m_t.cpu().numpy(), (w, h))
                            binary_mask[m_np > 0.5] = 255
                    
                    # 将单通道掩码应用到 RGB 图像上，实现抠图
                    # 结果为：有目标的地方保留原色，无目标的地方为黑色
                    extracted_mask_rgb = cv2.bitwise_and(cv_rgb, cv_rgb, mask=binary_mask)
                    # ---------------------------------------

                    # fn = f"frame_{saved_count:05d}.png"
                    fn = f"{saved_count}.png"
                    cv2.imwrite(os.path.join(dirs["rgb"], fn), cv_rgb)
                    cv2.imwrite(os.path.join(dirs["depth"], fn), cv_depth)
                    # 保存扣取后的三通道 RGB 图像
                    cv2.imwrite(os.path.join(dirs["mask"], fn), extracted_mask_rgb)
                    saved_count += 1
                    print(f"进度: {fn} | 同步误差: {diff:.4f}s", end='\r')
                except Exception as e:
                    print(f"\n[WARN] 处理错误: {e}")
                    continue

        bag.close()
        print(f"\n[SUCCESS] 完成！已保存 {saved_count} 组三通道数据至 {self.output_base}")

if __name__ == "__main__":
    processor = BagProcessor()
    processor.run()