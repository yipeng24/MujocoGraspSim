# DroneFullState.msg
# ==================
# 飞机全状态消息，由 mujoco_ros_bridge 在每个发布周期输出。
# 发布频率: ~125 Hz (每 4 个 MuJoCo 步，timestep=0.002s)
# 话题: /drone/full_state

Header header

# ================================================================
# 机体运动状态
# ================================================================

# 角速度 — body frame [rad/s]
# 来源: MuJoCo data.qvel[3:6] (freejoint, body frame)
geometry_msgs/Vector3 angular_velocity_body

# 线加速度 — world frame [m/s^2] (坐标加速度，含重力分量)
# 来源: MuJoCo data.qacc[0:3] (freejoint 平动分量)
# 注：若需 IMU proper acceleration，在 z 方向加 +9.81
geometry_msgs/Vector3 linear_acceleration_world

# 线速度 — world frame [m/s]
geometry_msgs/Vector3 linear_velocity_world

# 当前位置 — world frame [m]
geometry_msgs/Point position

# 当前姿态四元数 (w, x, y, z)
geometry_msgs/Quaternion current_attitude

# ================================================================
# 惯性参数（MuJoCo model 参数，随 set_payload 服务实时更新）
# ================================================================

# 飞机 base_link 主惯量轴转动惯量 — body principal frame [kg·m²]
# 来源: model.body_inertia[drone_body_id] (不含 payload 子树)
float64 ixx
float64 iyy
float64 izz

# 飞机子树总质量（含 payload）[kg]
# 来源: data.cinert[drone_body_id][9] (MuJoCo 每步自动计算)
float64 mass

# ================================================================
# Px4Ctrl 控制指令
# ================================================================

# 归一化升力 [0, 1] — Px4Ctrl 发出的 AttitudeTarget.thrust
float64 thrust_normalized

# 期望姿态四元数 (w, x, y, z) — Px4Ctrl 发出的目标姿态
geometry_msgs/Quaternion desired_attitude

# ================================================================
# PD 控制器输出（mujoco_bridge 内部 xfrc_applied）
# ================================================================

# 施加到机体的力 — world frame [N]
# 来源: data.xfrc_applied[drone_body_id][0:3]，包含 thrust 向量
geometry_msgs/Vector3 applied_force

# 施加到机体的力矩 — world frame [N·m]
# 来源: data.xfrc_applied[drone_body_id][3:6]，由姿态 PD 控制器计算
geometry_msgs/Vector3 applied_torque
