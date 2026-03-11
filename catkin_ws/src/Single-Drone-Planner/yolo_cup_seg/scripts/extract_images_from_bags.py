#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import rosbag
import sys
import glob

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
        self.bag_input_path = self.cfg.get("bag_settings.path", "")
        self.output_base = self.cfg.get("bag_settings.output_dir", "./output")
        self.extract_mask = self.cfg.get("bag_settings.extract_mask", False)
        self.frame_interval = int(self.cfg.get("bag_settings.frame_interval", 1))
        self.rgb_topic = self.cfg.get("topics.rgb", "/camera/image/compressed")
        self.depth_topic = self.cfg.get("topics.depth", "/camera/depth/image_raw")
        self.time_threshold = float(self.cfg.get("sync.time_threshold", 0.05))

        # 根据是否需要mask来决定是否加载YOLO
        self.model = None
        if self.extract_mask:
            # 条件导入 YOLO 和 torch
            from ultralytics import YOLO
            import torch
            
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
            print(f"[INFO] YOLO 模型已加载，将提取 mask")
            
            # Mask过滤参数
            self.min_mask_area = int(self.cfg.get("bag_settings.min_mask_area", 1000))
            self.keep_largest_component = self.cfg.get("bag_settings.keep_largest_component", True)
            print(f"[INFO] Mask过滤设置: 最小面积={self.min_mask_area}像素, 仅保留最大连通域={self.keep_largest_component}")
        else:
            print(f"[INFO] 跳过 YOLO 加载，仅提取 RGB 和 Depth")
        
        print(f"[INFO] 帧间隔设置: 每 {self.frame_interval} 帧提取一次")
        
        # 全局计数器，用于增量式命名
        self.global_saved_count = 0

    def _parse_fg_ids(self, param):
        if str(param).lower() in ("auto", "", "none"): return None
        try:
            if isinstance(param, list): return [int(x) for x in param]
            return [int(x) for x in str(param).split(",")]
        except: return None

    def filter_mask(self, binary_mask):
        """
        过滤mask：只保留最大连通域，并检查面积
        返回: (filtered_mask, is_valid)
        """
        if binary_mask is None or binary_mask.max() == 0:
            return binary_mask, False
        
        # 如果需要只保留最大连通域
        if self.keep_largest_component:
            # 查找连通域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            if num_labels <= 1:  # 只有背景，没有前景
                return binary_mask, False
            
            # 找到最大的连通域（排除背景0）
            areas = stats[1:, cv2.CC_STAT_AREA]  # 排除背景
            largest_label = np.argmax(areas) + 1  # +1因为排除了背景
            
            # 创建只包含最大连通域的mask
            filtered_mask = np.zeros_like(binary_mask)
            filtered_mask[labels == largest_label] = 255
            
            # 检查最大连通域的面积
            largest_area = areas[largest_label - 1]
        else:
            filtered_mask = binary_mask
            largest_area = np.sum(binary_mask > 0)
        
        # 判断面积是否满足要求
        is_valid = largest_area >= self.min_mask_area
        
        return filtered_mask, is_valid


    def process_single_bag(self, bag_file, dirs):
        """处理单个 Bag 文件的逻辑"""
        print(f"\n[PROCESS] 正在处理: {os.path.basename(bag_file)}")
        try:
            bag = rosbag.Bag(bag_file, 'r')
        except Exception as e:
            print(f"[ERROR] 无法读取 Bag: {e}")
            return

        # 索引深度图
        depth_msgs = []
        for _, msg, _ in bag.read_messages(topics=[self.depth_topic]):
            depth_msgs.append((msg.header.stamp.to_sec(), msg))
        
        if not depth_msgs:
            print(f"[WARN] {bag_file} 中未找到深度图数据!")
            bag.close()
            return
        
        depth_stamps = np.array([m[0] for m in depth_msgs])

        # 帧计数器，用于间隔提取
        frame_counter = 0
        
        for _, msg, _ in bag.read_messages(topics=[self.rgb_topic]):
            # 帧间隔过滤
            frame_counter += 1
            if frame_counter % self.frame_interval != 0:
                continue
            
            rgb_t = msg.header.stamp.to_sec()
            idx = np.argmin(np.abs(depth_stamps - rgb_t))
            diff = abs(depth_stamps[idx] - rgb_t)

            if diff < self.time_threshold:
                try:
                    if "CompressedImage" in msg._type:
                        cv_rgb = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                    else:
                        cv_rgb = fast_numpify_rgb(msg)

                    cv_depth = fast_numpify_depth(depth_msgs[idx][1])

                    # 条件处理mask
                    mask_is_valid = True  # 默认有效
                    if self.extract_mask and self.model is not None:
                        res = self.model.predict(cv_rgb, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
                        
                        h, w = cv_rgb.shape[:2]
                        binary_mask = np.zeros((h, w), dtype=np.uint8)
                        
                        if res.masks is not None:
                            clss = res.boxes.cls.cpu().numpy().astype(int)
                            for i, m_t in enumerate(res.masks.data):
                                if self.fg_class_ids and clss[i] not in self.fg_class_ids: 
                                    continue
                                m_np = cv2.resize(m_t.cpu().numpy(), (w, h))
                                binary_mask[m_np > 0.5] = 255
                        
                        # 过滤mask：只保留最大连通域并检查面积
                        binary_mask, mask_is_valid = self.filter_mask(binary_mask)
                        
                        if mask_is_valid:
                            extracted_mask_rgb = cv2.bitwise_and(cv_rgb, cv_rgb, mask=binary_mask)
                        else:
                            # Mask无效，跳过此帧的所有保存
                            continue

                    # 只有到这里才保存（mask有效或不需要mask）
                    # 使用全局计数器命名，保证连续
                    fn = f"{self.global_saved_count}.png"
                    
                    cv2.imwrite(os.path.join(dirs["rgb"], fn), cv_rgb)
                    cv2.imwrite(os.path.join(dirs["depth"], fn), cv_depth)
                    
                    # 只有在需要mask时才保存mask
                    if self.extract_mask and self.model is not None:
                        cv2.imwrite(os.path.join(dirs["mask"], fn), extracted_mask_rgb)
                    
                    # 计数器递增（只有保存时才递增）
                    self.global_saved_count += 1
                    print(f"进度: {fn} | 同步误差: {diff:.4f}s", end='\r')
                except Exception as e:
                    print(f"\n[WARN] 处理错误: {e}")
                    continue

        bag.close()

    def run(self):
        # 准备输出目录
        dirs = {
            "rgb": os.path.join(self.output_base, "rgb"),
            "depth": os.path.join(self.output_base, "depth")
        }
        
        # 只有在需要mask时才创建mask目录
        if self.extract_mask:
            dirs["mask"] = os.path.join(self.output_base, "mask")
        
        for d in dirs.values():
            if not os.path.exists(d): os.makedirs(d)

        # 判断输入类型：列表、文件夹或单个文件
        if isinstance(self.bag_input_path, list):
            # 模式1: 列表模式 - 配置文件中直接指定多个bag文件路径
            bag_files = []
            for path in self.bag_input_path:
                if os.path.isfile(path):
                    bag_files.append(path)
                else:
                    print(f"[WARN] 列表中的路径不存在或不是文件: {path}")
            bag_files = sorted(bag_files)
            print(f"[INFO] 列表模式: 找到 {len(bag_files)} 个 Bag 文件")
        elif os.path.isdir(self.bag_input_path):
            # 模式2: 文件夹模式 - 处理文件夹中所有.bag文件
            bag_files = sorted(glob.glob(os.path.join(self.bag_input_path, "*.bag")))
            print(f"[INFO] 文件夹模式: 找到 {len(bag_files)} 个 Bag 文件")
        elif os.path.isfile(self.bag_input_path):
            # 模式3: 单文件模式
            bag_files = [self.bag_input_path]
            print(f"[INFO] 单文件模式")
        else:
            print(f"[ERROR] 输入路径不存在: {self.bag_input_path}")
            return

        if not bag_files:
            print("[WARN] 未找到任何待处理的 .bag 文件")
            return

        # 遍历处理所有 Bag
        for bag_file in bag_files:
            self.process_single_bag(bag_file, dirs)

        print(f"\n[SUCCESS] 全部完成！共保存 {self.global_saved_count} 组数据至 {self.output_base}")

if __name__ == "__main__":
    processor = BagProcessor()
    processor.run()