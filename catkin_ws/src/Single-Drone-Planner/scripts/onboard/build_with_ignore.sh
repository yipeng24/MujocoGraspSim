#!/usr/bin/env bash
#
# 在 catkin_ws 根目录下运行（有 src/ 的那一层）
#
# 用法：
#   chmod +x catkin_ignore_folder_recursive.sh
#   ./catkin_ignore_folder_recursive.sh src/poscmd_modules
#
# 会递归查找目录下所有包含 package.xml 的目录，
# 创建 CATKIN_IGNORE -> catkin build -> 恢复。

set -e

WS_ROOT="$(pwd)"
TARGET_DIR="/home/nv/airgrasp_ws/src/airgrasp_planning/Single-Drone-Planner/uav_simulator"

# 1. 检查输入参数
if [[ -z "$TARGET_DIR" ]]; then
  echo "❌ 用法： $0 <folder>"
  echo "例如： $0 src/poscmd_modules"
  exit 1
fi

# 2. 转换为绝对路径
if [[ "$TARGET_DIR" != /* ]]; then
  TARGET_DIR="$WS_ROOT/$TARGET_DIR"
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "❌ 目录不存在: $TARGET_DIR"
  exit 1
fi

echo "🔧 Workspace: $WS_ROOT"
echo "📁 Target folder (递归忽略): $TARGET_DIR"
echo

# 3. 收集所有包含 package.xml 的目录 (使用 dirname 替代 :h)
# 使用 sort -u 进行去重
PKG_DIRS=($(find "$TARGET_DIR" -type f -name "package.xml" -exec dirname {} \; | sort -u))

# 4. 检查是否找到包
if [[ ${#PKG_DIRS[@]} -eq 0 ]]; then
  echo "⚠️ 在 $TARGET_DIR 下没有发现任何包含 package.xml 的目录，直接 catkin build"
  echo
  catkin build
  exit 0
fi

echo "🚫 将忽略以下 package 目录："
for d in "${PKG_DIRS[@]}"; do
  echo "  - $d"
done

# 5. 创建 CATKIN_IGNORE
echo
echo "📌 创建 CATKIN_IGNORE ..."
for d in "${PKG_DIRS[@]}"; do
  touch "$d/CATKIN_IGNORE"
done

# 6. 执行编译
# 使用 trap 确保即使编译失败，也能清理掉 CATKIN_IGNORE 文件
cleanup() {
  echo
  echo "🧹 正在清理 CATKIN_IGNORE 文件..."
  for d in "${PKG_DIRS[@]}"; do
    rm -f "$d/CATKIN_IGNORE"
  done
  echo "✅ 清理完成。"
}

# 注册钩子：脚本退出时（无论成功失败）都执行 cleanup
trap cleanup EXIT

echo
echo "🔨 Running catkin build ..."
catkin build

echo
echo "✅ 编译流程结束。"