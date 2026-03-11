#!/usr/bin/env python3
"""
mujoco_ros_bridge.py
====================
MuJoCo physics bridge for px4ctrl simulation.
Orchestrates the sub-modules in bridge/:

  bridge/physics.py       - DronePhysics  (apply_control, thrust/attitude state)
  bridge/viewer_vis.py    - ViewerVis     (MuJoCo user_scn rendering)
  bridge/planning_gate.py - PlanningGate  (G/H/F keys, command gate)
"""

import sys, os
# bridge/ is in the same directory as this script (scripts/)
sys.path.insert(0, os.path.dirname(__file__))

import rospy
import numpy as np
import mujoco
import mujoco.viewer
from threading import Lock

from nav_msgs.msg import Odometry, Path
from mavros_msgs.msg import AttitudeTarget
from sensor_msgs.msg import JointState
from quadrotor_msgs.msg import PositionCommand
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from bridge.physics       import DronePhysics
from bridge.viewer_vis    import ViewerVis
from bridge.planning_gate import PlanningGate

# drone_logger: 全状态发布 + payload 服务（可选，未编译时自动跳过）
from typing import Any
DroneFullState:     Any = None
SetPayload:         Any = None
SetPayloadResponse: Any = None
_LOGGER_AVAILABLE = False
try:
    from drone_logger.msg import DroneFullState       # type: ignore[import]
    from drone_logger.srv import SetPayload, SetPayloadResponse  # type: ignore[import]
    _LOGGER_AVAILABLE = True
except ImportError:
    pass


class MuJoCoROSBridge:
    def __init__(self):
        rospy.init_node('mujoco_ros_bridge', anonymous=True)

       
        model_path = rospy.get_param(
            '~model_path',
            '/home/lambyeeh/program/mujoco_ros1_docker/catkin_ws/src/drone_urdf/scene_drone.xml')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        self.lock  = Lock()

        self.drone_body_id = -1
        for name in ["base_link", "drone", "quadrotor"]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                self.drone_body_id = bid
                break
        if self.drone_body_id == -1:
            rospy.logwarn("Drone body not found by name, using id=1")
            self.drone_body_id = 1

        # 只用飞机自身子树的质量（排除场景里的 cup、obstacle_box 等自由体）
        self._drone_mass = self._subtree_mass(self.drone_body_id)
        rospy.loginfo(f"Drone subtree mass: {self._drone_mass:.3f} kg  "
                      f"(scene total: {float(np.sum(self.model.body_mass)):.3f} kg)")


        ARM_JOINT_NAMES = ["arm_pitch_joint", "arm_pitch2_joint", "arm_roll_joint"]
        arm_qadr, arm_vadr = [], []
        for jname in ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                arm_qadr.append(int(self.model.jnt_qposadr[jid]))
                arm_vadr.append(int(self.model.jnt_dofadr[jid]))
            else:
                rospy.logwarn(f"Arm joint '{jname}' not found in model")
                arm_qadr.append(-1)
                arm_vadr.append(-1)

        
        self.physics = DronePhysics(
            self.model, self.data, self.drone_body_id, arm_qadr, arm_vadr, self.lock)

        self.vis  = ViewerVis(self.lock)

        planning_mode = rospy.get_param('~planning_mode', False)
        self.gate = PlanningGate(self.lock, planning_mode)

        
        rospy.Subscriber('/attitude_cmd',    AttitudeTarget,  self._attitude_cb,        queue_size=1)
        rospy.Subscriber('/joint_state_cmd', JointState,      self._joint_cmd_cb,       queue_size=1)
        rospy.Subscriber('/arm_grab_cmd',    Bool,            self._grab_cmd_cb,        queue_size=1)
        rospy.Subscriber('/planning/astar',  Path,            self.vis.set_astar_path,  queue_size=1)
        rospy.Subscriber('/planning/traj',   Path,            self.vis.set_opt_traj,    queue_size=1)
        rospy.Subscriber('/position_cmd',     PositionCommand, self._position_cmd_cb,   queue_size=1)
        rospy.Subscriber('/goal_preview',     PoseStamped,     self._goal_picker_cb,   queue_size=1)
        rospy.Subscriber('/goal_arm_preview', JointState,      self._goal_arm_vis_cb,  queue_size=1)

        self.odom_pub         = rospy.Publisher('/odom',         Odometry,   queue_size=10)
        self.visual_odom_pub  = rospy.Publisher('/visual_odom',  Odometry,   queue_size=10)
        self.joint_states_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)

        # ── payload body（仅 scene_drone_payload.xml 有此 body） ──────────────
        self._payload_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
        if self._payload_body_id != -1:
            rospy.loginfo(f"[bridge] Payload body found: id={self._payload_body_id}  "
                          f"initial mass={float(self.model.body_mass[self._payload_body_id]):.3f} kg")
        else:
            rospy.loginfo("[bridge] No payload body in model (running without payload)")

        # ── 全状态发布（drone_logger 包可用时） ───────────────────────────────
        self._last_thrust_normalized = 0.0
        self._last_desired_quat      = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_xfrc              = np.zeros(6)
        self._pending_payload        = None   # (mass, cx, cy, cz, enable)

        if _LOGGER_AVAILABLE:
            self.full_state_pub = rospy.Publisher(
                '/drone/full_state', DroneFullState, queue_size=10)
            rospy.loginfo("[bridge] DroneFullState publisher ready → /drone/full_state")
            if self._payload_body_id != -1:
                rospy.Service('/drone/set_payload', SetPayload, self._srv_set_payload)
                rospy.loginfo("[bridge] SetPayload service ready → /drone/set_payload")
        else:
            rospy.logwarn("[bridge] drone_logger not built — full state pub + payload service disabled. "
                          "Run: cd catkin_ws && catkin_make --only-pkg-with-deps drone_logger")

        self.dt               = self.model.opt.timestep
        self.rate             = rospy.Rate(500)
        self.sim_step_counter = 0
        self._viewer_ref      = None
        self._last_arm        = np.zeros(3)
        self._last_odom_pos   = None
        self._last_odom_time  = None
        self._last_lin_vel    = np.zeros(3)

        rospy.loginfo(f"MuJoCo ROS Bridge initialized | model: {model_path}")
        rospy.loginfo(f"Drone body ID: {self.drone_body_id} | timestep: {self.dt}s")
        if planning_mode:
            rospy.loginfo("Keys: G = set planning goal | F = approve & fly | H = hover")
        else:
            rospy.loginfo("Keys: G = set goal at lookat XY | H = hover in place")

    def _subtree_mass(self, body_id: int) -> float:
        """递归累加 body_id 及其所有子 body 的质量（飞机子树，不含 cup/obstacle 等自由体）。"""
        total = float(self.model.body_mass[body_id])
        for child_id in range(self.model.nbody):
            if self.model.body_parentid[child_id] == body_id:
                total += self._subtree_mass(child_id)
        return total

    # ──────────────────────────────────────────────────────────────────────────
    # Payload service（仅在 scene_drone_payload.xml 中有 payload body 时注册）
    # ──────────────────────────────────────────────────────────────────────────

    def _srv_set_payload(self, req):
        """
        SetPayload service callback.
        修改 payload body 的质量和质心（model 参数），下一个 mj_step 立即生效。
        """
        if not _LOGGER_AVAILABLE:
            return SetPayloadResponse(success=False, message="drone_logger not available",
                                      total_mass=0.0)
        if self._payload_body_id == -1:
            return SetPayloadResponse(success=False,
                                      message="No payload body in model (use use_payload:=true)",
                                      total_mass=self._subtree_mass(self.drone_body_id))
        # 挂起到主循环中安全执行（Python 引用赋值是原子的）
        self._pending_payload = (req.mass, req.com_x, req.com_y, req.com_z, req.enable)

        # 等待主循环消费（最多 0.1 s）
        import time as _time
        for _ in range(100):
            if self._pending_payload is None:
                break
            _time.sleep(0.001)

        total = float(self.data.cinert[self.drone_body_id][9])
        if req.enable:
            msg = (f"Payload ON: mass={req.mass:.3f} kg, "
                   f"com=({req.com_x:.3f}, {req.com_y:.3f}, {req.com_z:.3f}) m")
        else:
            msg = "Payload OFF: mass set to 0"
        rospy.loginfo(f"[payload] {msg}  total={total:.3f} kg")
        return SetPayloadResponse(success=True, message=msg, total_mass=total)

    def _apply_pending_payload(self):
        """在主仿真循环中安全地将 pending payload 参数写入 model（线程安全）。"""
        if self._pending_payload is None:
            return
        mass, cx, cy, cz, enable = self._pending_payload
        self._pending_payload = None   # 先清除标志（原子赋值）

        bid = self._payload_body_id
        if enable:
            new_mass = max(0.0, float(mass))
            self.model.body_mass[bid] = new_mass
            self.model.body_ipos[bid] = [float(cx), float(cy), float(cz)]
            # 按均质方块公式更新转动惯量：I = m*(a^2+b^2)/3，半边长 a=0.025m
            a = 0.025
            I = new_mass * 2 * a * a / 3.0
            self.model.body_inertia[bid] = [I, I, I]
        else:
            self.model.body_mass[bid] = 0.0
            self.model.body_inertia[bid] = [1e-9, 1e-9, 1e-9]

    # ──────────────────────────────────────────────────────────────────────────
    # DroneFullState publisher
    # ──────────────────────────────────────────────────────────────────────────

    def _publish_full_state(self, stamp):
        """每个发布周期（~125 Hz）将飞机全状态打包为 DroneFullState 并发布。"""
        if not _LOGGER_AVAILABLE or not hasattr(self, 'full_state_pub'):
            return

        fs = DroneFullState()
        fs.header.stamp    = stamp
        fs.header.frame_id = "world"

        # 机体角速度（body frame，freejoint qvel[3:6]）
        fs.angular_velocity_body.x = float(self.data.qvel[3])
        fs.angular_velocity_body.y = float(self.data.qvel[4])
        fs.angular_velocity_body.z = float(self.data.qvel[5])

        # 线加速度（world frame，freejoint qacc[0:3]，含重力分量）
        fs.linear_acceleration_world.x = float(self.data.qacc[0])
        fs.linear_acceleration_world.y = float(self.data.qacc[1])
        fs.linear_acceleration_world.z = float(self.data.qacc[2])

        # 线速度（world frame）
        fs.linear_velocity_world.x = float(self._last_lin_vel[0])
        fs.linear_velocity_world.y = float(self._last_lin_vel[1])
        fs.linear_velocity_world.z = float(self._last_lin_vel[2])

        # 位置（world frame）
        pos = self.data.xpos[self.drone_body_id]
        fs.position.x = float(pos[0])
        fs.position.y = float(pos[1])
        fs.position.z = float(pos[2])

        # 当前姿态（world frame）
        quat = self.data.xquat[self.drone_body_id]
        fs.current_attitude.w = float(quat[0])
        fs.current_attitude.x = float(quat[1])
        fs.current_attitude.y = float(quat[2])
        fs.current_attitude.z = float(quat[3])

        # 转动惯量（model.body_inertia：body principal frame 对角元素）
        inertia = self.model.body_inertia[self.drone_body_id]
        fs.ixx = float(inertia[0])
        fs.iyy = float(inertia[1])
        fs.izz = float(inertia[2])

        # 飞机子树总质量（含 payload，由 MuJoCo mj_crb 每步计算）
        fs.mass = float(self.data.cinert[self.drone_body_id][9])

        # Px4Ctrl 输出
        with self.lock:
            fs.thrust_normalized = self._last_thrust_normalized
            q = self._last_desired_quat.copy()
        fs.desired_attitude.w = float(q[0])
        fs.desired_attitude.x = float(q[1])
        fs.desired_attitude.y = float(q[2])
        fs.desired_attitude.z = float(q[3])

        # PD 控制器输出（apply_control 后保存的 xfrc_applied）
        fs.applied_force.x  = float(self._last_xfrc[0])
        fs.applied_force.y  = float(self._last_xfrc[1])
        fs.applied_force.z  = float(self._last_xfrc[2])
        fs.applied_torque.x = float(self._last_xfrc[3])
        fs.applied_torque.y = float(self._last_xfrc[4])
        fs.applied_torque.z = float(self._last_xfrc[5])

        self.full_state_pub.publish(fs)

    def _attitude_cb(self, msg):
        """Receive AttitudeTarget from px4ctrl (via mavros_sim relay)."""
        IGNORE_ATTITUDE = 128
        with self.lock:
            self.physics.use_rate_ctrl = bool(msg.type_mask & IGNORE_ATTITUDE)
            self.physics.attitude_quat = np.array([
                msg.orientation.w, msg.orientation.x,
                msg.orientation.y, msg.orientation.z])
            self.physics.body_rate_des = np.array([
                msg.body_rate.x, msg.body_rate.y, msg.body_rate.z])
            mass = self._drone_mass
            self.physics.thrust        = msg.thrust * 2.0 * mass * 9.81
            self.physics.received_cmd  = True
            # 保存用于全状态发布
            self._last_thrust_normalized = float(msg.thrust)
            self._last_desired_quat = np.array([
                msg.orientation.w, msg.orientation.x,
                msg.orientation.y, msg.orientation.z])

    def _goal_picker_cb(self, msg):
        """Goal picker double-click → ghost appears at that position."""
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = float(np.arctan2(siny, cosy))
        self.vis.set_goal_state(pos, yaw, self._last_arm)

    def _position_cmd_cb(self, *_):
        """Ghost arm is now updated via /goal_arm_preview. This callback is kept as a stub."""
        pass

    def _goal_arm_vis_cb(self, msg):
        """Receive arm_target from joy_to_position_cmd → update ghost arm in viewer."""
        if len(msg.position) >= 3:
            arm = np.array(msg.position[:3])
            self._last_arm = arm
            self.vis.update_goal_arm(arm)

    def _joint_cmd_cb(self, msg):
        """Receive arm joint target angles from px4ctrl (3 joints: pitch, pitch2, roll)."""
        if len(msg.position) >= 3:
            self.physics.arm_target = np.array(msg.position[:3], dtype=float)

    def _grab_cmd_cb(self, msg):
        """Receive gripper command from px4ctrl (/arm_grab_cmd).
        Aligns with airgrasp: false=OPEN(0.025), true=CLOSE(-0.025).
        ctrlrange of arm_right_finger_joint: -0.025 ~ 0.025
        """
        self.physics.gripper_target = -0.025 if msg.data else 0.025

    

    def _get_odometry(self, stamp=None):
        odom                         = Odometry()
        odom.header.stamp            = rospy.Time.now() if stamp is None else stamp
        odom.header.frame_id         = "world"
        odom.child_frame_id          = "body"
        pos                          = self.data.xpos[self.drone_body_id]
        quat                         = self.data.xquat[self.drone_body_id]
        odom.pose.pose.position.x    = pos[0]
        odom.pose.pose.position.y    = pos[1]
        odom.pose.pose.position.z    = pos[2]
        odom.pose.pose.orientation.w = quat[0]
        odom.pose.pose.orientation.x = quat[1]
        odom.pose.pose.orientation.y = quat[2]
        odom.pose.pose.orientation.z = quat[3]
        if self._last_odom_time is not None:
            dt = (odom.header.stamp - self._last_odom_time).to_sec()
            if dt > 1e-6:
                self._last_lin_vel = (pos - self._last_odom_pos) / dt

        odom.twist.twist.linear.x    = float(self._last_lin_vel[0])
        odom.twist.twist.linear.y    = float(self._last_lin_vel[1])
        odom.twist.twist.linear.z    = float(self._last_lin_vel[2])
        odom.twist.twist.angular.x   = self.data.qvel[3]
        odom.twist.twist.angular.y   = self.data.qvel[4]
        odom.twist.twist.angular.z   = self.data.qvel[5]
        odom.pose.covariance         = [0.01] * 36
        odom.twist.covariance        = [0.01] * 36
        self._last_odom_pos          = pos.copy()
        self._last_odom_time         = odom.header.stamp
        return odom

    def _get_joint_states(self):
        js              = JointState()
        js.header.stamp = rospy.Time.now()
        js.name         = ["arm_pitch_joint", "arm_pitch2_joint", "arm_roll_joint"]
        js.position, js.velocity, js.effort = [], [], []
        for qadr, vadr in zip(self.physics.arm_qadr, self.physics.arm_vadr):
            if qadr != -1:
                js.position.append(float(self.data.qpos[qadr]))
                js.velocity.append(float(self.data.qvel[vadr]))
            else:
                js.position.append(0.0)
                js.velocity.append(0.0)
            js.effort.append(0.0)
        return js

    

    def _sim_step_and_publish(self):
        # 主循环中安全应用 pending payload 参数
        self._apply_pending_payload()

        self.physics.apply_control()

        # 保存 PD 控制器输出（apply_control 刚写完，mj_step 前读取）
        self._last_xfrc = self.data.xfrc_applied[self.drone_body_id].copy()

        mujoco.mj_step(self.model, self.data)
        self.sim_step_counter += 1

        if self.sim_step_counter % 4 == 0:
            t               = rospy.Time.now()
            odom            = self._get_odometry(t)
            self.odom_pub.publish(odom)
            self.visual_odom_pub.publish(odom)
            js              = self._get_joint_states()
            js.header.stamp = t
            self.joint_states_pub.publish(js)
            # 全状态发布（drone_logger 可用时）
            self._publish_full_state(t)

    
    def _viewer_key_callback(self, keycode: int):
        viewer = self._viewer_ref
        if viewer is None:
            return
        pos  = self.data.xpos[self.drone_body_id].copy()
        quat = self.data.xquat[self.drone_body_id]
        siny = 2.0 * (quat[0]*quat[3] + quat[1]*quat[2])
        cosy = 1.0 - 2.0 * (quat[2]*quat[2] + quat[3]*quat[3])
        yaw  = float(np.arctan2(siny, cosy))
        self.gate.handle_key(keycode, pos, yaw, viewer)

    

    def run(self):
        rospy.loginfo("Starting MuJoCo simulation...")

        if hasattr(self.model, 'jnt_qposadr'):
            self.data.qpos[self.model.jnt_qposadr[0] + 2] = 0.0
        mujoco.mj_forward(self.model, self.data)

        use_viewer = rospy.get_param('~use_viewer', False)

        if use_viewer:
            with mujoco.viewer.launch_passive(
                    self.model, self.data,
                    key_callback=self._viewer_key_callback) as viewer:
                self._viewer_ref = viewer
                rospy.loginfo("MuJoCo viewer launched")

                while not rospy.is_shutdown() and viewer.is_running():
                    self._sim_step_and_publish()
                    self.gate.publish_goal_if_set()

                    if self.sim_step_counter % 8 == 0:
                        drone_pos = self.data.xpos[self.drone_body_id].copy()
                        self.vis.update_trail(drone_pos)
                        self.vis.render(viewer.user_scn, drone_pos, self.gate.goal_pos)
                        viewer.sync()

                    self.rate.sleep()

                self._viewer_ref = None
        else:
            rospy.loginfo("Running headless (no viewer)")
            while not rospy.is_shutdown():
                self._sim_step_and_publish()
                self.rate.sleep()

        rospy.loginfo("MuJoCo simulation stopped")


if __name__ == '__main__':
    try:
        bridge = MuJoCoROSBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        pass
