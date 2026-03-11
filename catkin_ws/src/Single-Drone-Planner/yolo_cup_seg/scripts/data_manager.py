# -*- coding: utf-8 -*-
import yaml
import os
import rospy

class ConfigManager:
    def __init__(self, config_path=None):
        self.config = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path):
        """加载 YAML 配置文件"""
        rospy.loginfo(f"[ConfigManager] Loading config from: {config_path}")
        if not os.path.exists(config_path):
            rospy.logerr(f"[ConfigManager] Config file not found: {config_path}")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                rospy.loginfo("[ConfigManager] Config loaded successfully.")
        except Exception as e:
            rospy.logerr(f"[ConfigManager] Failed to load config: {e}")

    def get(self, key, default=None):
        """支持嵌套键获取，例如 'model.path'"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            # rospy.logwarn(f"[ConfigManager] Key '{key}' not found, using default: {default}")
            return default