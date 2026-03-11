#!/usr/bin/env python3
"""
joy_to_position_cmd.py
======================
将 /joy_free (key_to_joy_UAM.py 发布) 转换为 /position_cmd，
支持 MANUAL / PLANNING 两种模式。

键位（由 key_to_joy_UAM.py 捕获）:
  W/S/A/D    前/后/左/右（机体系相对）
  Q/E        上/下
  ←/→       偏航
  5 (MODE)   切换 MANUAL ↔ PLANNING 模式
  6 (GEAR)   就地悬停（hold current position）

MANUAL 模式:
  键盘 axes → 积分位置 → 发布 /position_cmd

PLANNING 模式:
  订阅 /position_cmd_planner → 透传到 /position_cmd
  键盘移动指令被屏蔽，让规划器控制无人机

发布: /ctrl_mode (std_msgs/String) → key_to_joy_UAM.py 读取并显示
"""

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy, JointState
from quadrotor_msgs.msg import PositionCommand, TakeoffLand, UAMFullState
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Point, Pose
from visualization_msgs.msg import MarkerArray, Marker
import tf.transformations as tft


# ── 正向运动学辅助（与 ch_rc_sdf/params.yaml 运动链一致）─────────────────────
def _mat4(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T

def _Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def _Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def _Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

# joy 轴索引（与 key_to_joy_UAM.py / input.h 一致）
CH_FORWARD = 0
CH_LEFT    = 1
CH_UP      = 2
CH_YAW     = 4
CH_PITCH   = 5
CH_PITCH2  = 6
CH_ROLL    = 7

BUTTON_CANCEL     = 0   # BACKSPACE → 解锁 / 降落
BUTTON_CONFIRM    = 1   # ENTER    → 起飞
BUTTON_GRAB_OPEN  = 2   # O        → 夹爪张开
BUTTON_GRAB_CLOSE = 3   # P        → 夹爪闭合
BUTTON_MODE       = 4   # key 5    → 切换 MANUAL / PLANNING
BUTTON_GEAR       = 5   # key 6    → 就地悬停
BUTTON_LOCK       = 6   # '        → 锁定当前目标位置

MODE_MANUAL   = 'MANUAL'
MODE_PLANNING = 'PLANNING'


class JoyToPositionCmd:
    def __init__(self):
        rospy.init_node('joy_to_position_cmd')

        self.vel          = rospy.get_param('~vel',          0.5)
        self.omega        = rospy.get_param('~omega',        0.5)
        self.deadzone     = rospy.get_param('~deadzone',     0.05)
        self.smooth_alpha = rospy.get_param('~smooth_alpha', 0.2)
        self.arm_speed    = rospy.get_param('~arm_speed',    1.0)
        self.rate_hz      = 50.0

        # ── 飞行状态 ──────────────────────────────────────────────────────
        self.target_pos  = None        # np.array [x, y, z]
        self.target_yaw  = 0.0
        self.has_odom    = False
        self.hover_ready = False       # True after z > 0.8 m
        self.drone_pos   = np.zeros(3)
        self.drone_yaw   = 0.0

        # ── 手臂 ──────────────────────────────────────────────────────────
        self.arm_target   = np.zeros(3)
        self.ARM_LO = np.array([0.0,       -np.pi/2, -0.8*np.pi])
        self.ARM_HI = np.array([np.pi/2,    np.pi/2,  0.8*np.pi])
        self._arm_hold        = np.zeros(3)  # 双击时锁定的臂角（hover保持，作为init_thetas）
        self._goal_picked     = False        # True=已双击选目标，键盘调臂只改ghost不动实际臂
        self._mission_active  = False        # True=已发送任务给规划器，臂角由规划器控制
        self._last_goal_pos   = None         # 上次双击的位置，用于判断是否新目标
        self._mode_switch_time = None        # 切换到PLANNING的时刻，用于过渡期位置保持
        # 规划器目标（与 hover target_pos 分开）— 仅在 CONFIRM 时使用
        self._planner_goal_pos = None        # 双击目标位置（规划器终点）
        self._planner_goal_yaw = 0.0         # 双击时无人机的偏航（规划器终点 yaw）

        # ── joy ───────────────────────────────────────────────────────────
        self.joy_axes    = [0.0] * 8
        self.smooth_axes = [0.0] * 8

        # ── 模式 ──────────────────────────────────────────────────────────
        self._mode        = MODE_MANUAL   # 当前飞行控制模式
        self._planner_cmd = None          # 最新规划器命令
        self._is_locked   = False         # LOCK 锁定目标位置标志

        # ── 按钮边沿检测（避免长按重复触发）────────────────────────────────
        self._last_lock    = 0
        self._last_confirm = 0
        self._last_cancel  = 0
        self._last_mode    = 0
        self._last_gear    = 0
        self._dbg_cnt      = 0
        self._dbg_planner_cnt = 0

        # ── ROS 接口 ──────────────────────────────────────────────────────
        self.odom_sub       = rospy.Subscriber('/odom',                  Odometry,        self._odom_cb,     queue_size=1)
        self.joy_sub        = rospy.Subscriber('/joy_free',              Joy,             self._joy_cb,      queue_size=1)
        self.planner_sub    = rospy.Subscriber('/position_cmd_planner',  PositionCommand, self._planner_cb,  queue_size=1)
        self.cmd_pub        = rospy.Publisher('/position_cmd',           PositionCommand, queue_size=1)
        self.takeoff_pub    = rospy.Publisher('/px4ctrl/takeoff_land',   TakeoffLand,     queue_size=1)
        self.mode_pub       = rospy.Publisher('/ctrl_mode',              String,          queue_size=1, latch=True)
        self.goal_pub        = rospy.Publisher('/uam_state_goal_cmd',    UAMFullState,    queue_size=1)
        self.goal_vis_pub    = rospy.Publisher('/uam_state_goal_vis',    MarkerArray,     queue_size=1)
        self.arm_vis_pub     = rospy.Publisher('/goal_arm_preview',      JointState,      queue_size=1)
        self.grab_pub        = rospy.Publisher('/arm_grab_cmd',          Bool,            queue_size=1)
        rospy.Subscriber('/goal_preview',          PoseStamped, self._goal_picked_cb, queue_size=1)

        self._publish_mode()
        rospy.loginfo('[joy_to_pos_cmd] Ready.')
        rospy.loginfo('[joy_to_pos_cmd]   5=MODE(MANUAL/PLANNING)  6=GEAR(Hover)')
        rospy.loginfo("[joy_to_pos_cmd]   '=LOCK  ENTER=起飞  BACKSPACE=降落/解锁")
        rospy.loginfo('[joy_to_pos_cmd] 等待无人机起飞到 0.8m 以上后键盘控制激活...')

    # ── 回调 ──────────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        self.drone_pos[:] = [p.x, p.y, p.z]
        self.drone_yaw    = float(np.arctan2(siny, cosy))
        self.has_odom     = True

        # 无人机到达悬停高度后才初始化目标（避免起飞前误发 position_cmd）
        if not self.hover_ready and p.z > 0.8:
            self.target_pos = self.drone_pos.copy()
            self.target_yaw = self.drone_yaw
            self.hover_ready = True
            rospy.loginfo('[joy_to_pos_cmd] 无人机已悬停，键盘控制激活！')

    def _joy_cb(self, msg):
        n = len(msg.axes)
        for i in range(min(8, n)):
            self.joy_axes[i] = msg.axes[i]

        btn = msg.buttons

        def _btn(i):
            return len(btn) > i and btn[i] == 1

        # ── LOCK (' 键，button[6]) — 边沿触发，锁定/解锁目标位置 ─────────
        cur_lock = 1 if _btn(BUTTON_LOCK) else 0
        if cur_lock and not self._last_lock:
            if not self._is_locked:
                self._is_locked = True
                rospy.loginfo(f'[joy_to_pos_cmd] LOCKED at '
                              f'({self.target_pos[0]:.2f}, {self.target_pos[1]:.2f}, '
                              f'{self.target_pos[2]:.2f}) — 键盘移动暂停')
            else:
                self._is_locked = False
                rospy.loginfo('[joy_to_pos_cmd] UNLOCKED — 键盘移动恢复')
            self._publish_mode()
        self._last_lock = cur_lock

        # ── CONFIRM (ENTER，button[1]) — 边沿触发 ────────────────────────
        # 未起飞 → 发送起飞指令；已起飞 → 发送当前目标位姿给规划器
        cur_confirm = 1 if _btn(BUTTON_CONFIRM) else 0
        if cur_confirm and not self._last_confirm:
            if not self.hover_ready:
                tl = TakeoffLand()
                tl.takeoff_land_cmd = TakeoffLand.TAKEOFF
                self.takeoff_pub.publish(tl)
                rospy.loginfo('[joy_to_pos_cmd] CONFIRM → 发送起飞指令')
            elif not self._goal_picked or self._planner_goal_pos is None:
                rospy.logwarn('[joy_to_pos_cmd] CONFIRM 忽略：尚未双击选取目标点')
            else:
                # 发送 UAMFullState 到规划器（完全对齐 airgrasp joy2UAM run_mode=1）
                # 规划器从 /joint_states 读取 init_thetas（当前臂角），从此消息读取 final_thetas
                # 生成 WITHYAWANDTHETA 轨迹，臂角由规划器在轨迹中插值
                # 使用 _planner_goal_pos（双击目标），不用 target_pos（hover 位置）
                _, _, _, p_ee = self._fk_goal()
                q = tft.quaternion_from_euler(0.0, 0.0, float(self._planner_goal_yaw))
                goal = UAMFullState()
                goal.header.stamp    = rospy.Time.now()
                goal.header.frame_id = 'world'
                goal.pose.position.x = float(self._planner_goal_pos[0])
                goal.pose.position.y = float(self._planner_goal_pos[1])
                goal.pose.position.z = float(self._planner_goal_pos[2])
                goal.pose.orientation.x = float(q[0])
                goal.pose.orientation.y = float(q[1])
                goal.pose.orientation.z = float(q[2])
                goal.pose.orientation.w = float(q[3])
                goal.theta  = self.arm_target.tolist()
                goal.dtheta = [0.0, 0.0, 0.0]
                goal.ee_pose.position.x = float(p_ee[0])
                goal.ee_pose.position.y = float(p_ee[1])
                goal.ee_pose.position.z = float(p_ee[2])
                goal.ee_pose.orientation.w = 1.0
                self.goal_pub.publish(goal)
                self._mission_active = True
                rospy.loginfo(
                    f'[joy_to_pos_cmd] CONFIRM → MISSION SENT (UAMFullState)! '
                    f'goal=({self._planner_goal_pos[0]:.2f},{self._planner_goal_pos[1]:.2f},{self._planner_goal_pos[2]:.2f}) '
                    f'yaw={self._planner_goal_yaw:.2f} '
                    f'arm_goal={[round(float(v),2) for v in self.arm_target]}'
                )
        self._last_confirm = cur_confirm

        # ── CANCEL (BACKSPACE，button[0]) — 边沿触发，解锁或降落 ─────────
        cur_cancel = 1 if _btn(BUTTON_CANCEL) else 0
        if cur_cancel and not self._last_cancel:
            if self._is_locked:
                self._is_locked = False
                rospy.loginfo('[joy_to_pos_cmd] CANCEL → UNLOCKED')
            else:
                tl = TakeoffLand()
                tl.takeoff_land_cmd = TakeoffLand.LAND
                self.takeoff_pub.publish(tl)
                rospy.loginfo('[joy_to_pos_cmd] CANCEL → 发送降落指令')
        self._last_cancel = cur_cancel

        # ── MODE (key 5，button[4]) — 边沿触发，切换 MANUAL / PLANNING
        cur_mode = 1 if _btn(BUTTON_MODE) else 0
        if cur_mode and not self._last_mode:
            if self._mode == MODE_MANUAL:
                self._mode = MODE_PLANNING
                self._mode_switch_time = rospy.Time.now()
                rospy.loginfo('[joy_to_pos_cmd] >>> MODE: 切换为 PLANNING 模式 <<<')
            else:
                self._mode = MODE_MANUAL
                if self.has_odom:
                    self.target_pos = self.drone_pos.copy()
                    self.target_yaw = self.drone_yaw
                self._goal_picked      = False
                self._mission_active   = False
                self._last_goal_pos    = None
                self._planner_goal_pos = None
                rospy.loginfo('[joy_to_pos_cmd] >>> MODE: 切换为 MANUAL 模式 <<<')
            self._publish_mode()
        self._last_mode = cur_mode

        # ── GEAR (key 6，button[5]) — 边沿触发，就地悬停
        cur_gear = 1 if _btn(BUTTON_GEAR) else 0
        if cur_gear and not self._last_gear:
            if self.has_odom:
                self.target_pos = self.drone_pos.copy()
                self.target_yaw = self.drone_yaw
                self.smooth_axes = [0.0] * 8
                self._is_locked = False
                self._publish_mode()
                rospy.loginfo(f'[joy_to_pos_cmd] GEAR 就地悬停: ({self.drone_pos[0]:.2f}, '
                              f'{self.drone_pos[1]:.2f}, {self.drone_pos[2]:.2f})')
        self._last_gear = cur_gear

        # ── GRAB (O=张开, P=闭合，button[2]/[3]) — 直接转发到 /arm_grab_cmd ──
        if _btn(BUTTON_GRAB_OPEN):
            self.grab_pub.publish(Bool(data=False))
        elif _btn(BUTTON_GRAB_CLOSE):
            self.grab_pub.publish(Bool(data=True))

    def _planner_cb(self, msg):
        self._planner_cmd = msg
        # grab_cmd 只发一帧，50Hz 循环会错过，在回调里立即转发
        if msg.grab_cmd != 2:
            self.grab_pub.publish(Bool(data=(msg.grab_cmd == 1)))
        self._dbg_planner_cnt += 1
        if self._dbg_planner_cnt == 1 or self._dbg_planner_cnt % 50 == 0:
            th = list(msg.theta.position) if msg.theta.position else []
            rospy.logwarn('[DBG planner_cb #%d] flag=%d pos=(%.2f,%.2f,%.2f) theta=%s',
                self._dbg_planner_cnt, msg.trajectory_flag,
                msg.position.x, msg.position.y, msg.position.z,
                [round(v, 3) for v in th])

    # ── 工具 ──────────────────────────────────────────────────────────────

    def _publish_mode(self):
        s = self._mode
        if self._is_locked:
            s += '|LOCKED'
        self.mode_pub.publish(String(data=s))

    def _dz(self, v):
        return v if abs(v) > self.deadzone else 0.0

    # ── 目标位姿可视化（仿 airgrasp ch_rc_sdf uam_state_goal_vis）────────

    def _fk_goal(self):
        """正向运动学：计算目标状态下各关节/末端在世界系的位置。
        运动链参数来自 ch_rc_sdf/params.yaml：
          box0 → box1: T=(-0.015, 0.001, -0.096), Ry(theta[0])
          box1 → box2: T=(0.15, -0.0012, 0.0),   Ry(theta[1])
          box2 → box3: T=(0.12,  0.0,    0.0006), Rx(theta[2])
        返回: (p_body, p_j1, p_j2, p_ee) — 世界系坐标
        """
        th = self.arm_target
        T0 = _mat4(_Rz(self.target_yaw), self.target_pos)
        T1 = T0 @ _mat4(_Ry(th[0]), np.array([-0.015,  0.001, -0.096]))
        T2 = T1 @ _mat4(_Ry(th[1]), np.array([ 0.150, -0.0012,  0.0  ]))
        T3 = T2 @ _mat4(_Rx(th[2]), np.array([ 0.120,  0.0,     0.0006]))
        return self.target_pos.copy(), T1[:3, 3], T2[:3, 3], T3[:3, 3]

    def _publish_goal_vis(self):
        """发布目标位姿的 MarkerArray，在 RViz 中显示"幽灵"无人机+机械臂。"""
        if not self.hover_ready or self.target_pos is None:
            return

        p0, p1, p2, p3 = self._fk_goal()
        t = rospy.Time.now()
        ma = MarkerArray()

        def _pt(p):
            pt = Point(); pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2]); return pt

        def _color(m, r, g, b, a):
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a

        def _hdr(m, ns, mid, mtype):
            m.header.stamp = t; m.header.frame_id = 'world'
            m.ns = ns; m.id = mid; m.type = mtype; m.action = Marker.ADD

        # — 机身圆盘（flat cylinder）
        m = Marker(); _hdr(m, 'goal_body', 0, Marker.CYLINDER)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(p0[0]), float(p0[1]), float(p0[2])
        q = tft.quaternion_from_euler(0.0, 0.0, float(self.target_yaw))
        m.pose.orientation.x, m.pose.orientation.y = q[0], q[1]
        m.pose.orientation.z, m.pose.orientation.w = q[2], q[3]
        m.scale.x, m.scale.y, m.scale.z = 0.34, 0.34, 0.04
        _color(m, 0.2, 0.8, 0.2, 0.45)
        ma.markers.append(m)

        # — 机械臂连杆（LINE_STRIP，3段：机身→关节1→关节2→末端）
        for i, (pa, pb, r, g, b) in enumerate([
            (p0, p1, 1.0, 0.55, 0.0),   # 机身 → 关节1
            (p1, p2, 1.0, 0.75, 0.0),   # 连杆1
            (p2, p3, 0.9, 0.9,  0.0),   # 连杆2
        ]):
            m = Marker(); _hdr(m, 'goal_arm', i, Marker.LINE_STRIP)
            m.pose.orientation.w = 1.0
            m.scale.x = 0.025
            _color(m, r, g, b, 0.85)
            m.points.append(_pt(pa)); m.points.append(_pt(pb))
            ma.markers.append(m)

        # — 末端执行器球
        m = Marker(); _hdr(m, 'goal_ee', 0, Marker.SPHERE)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(p3[0]), float(p3[1]), float(p3[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.06
        _color(m, 1.0, 0.2, 0.2, 0.9)
        ma.markers.append(m)

        self.goal_vis_pub.publish(ma)

        # 发布 arm_target 供 MuJoCo ghost 使用
        js = JointState()
        js.header.stamp = t
        js.name     = ['arm_pitch_joint', 'arm_pitch2_joint', 'arm_roll_joint']
        js.position = self.arm_target.tolist()
        js.velocity = [0.0, 0.0, 0.0]
        self.arm_vis_pub.publish(js)

    # ── 主循环 ────────────────────────────────────────────────────────────

    def run(self):
        rate    = rospy.Rate(self.rate_hz)
        dt      = 1.0 / self.rate_hz
        vis_cnt = 0

        while not rospy.is_shutdown():
            if not self.hover_ready:
                rate.sleep()
                continue

            if self._mode == MODE_MANUAL:
                self._run_manual(dt)
            else:
                self._run_planning()

            vis_cnt += 1
            if vis_cnt % 5 == 0:   # 10 Hz
                self._publish_goal_vis()

            rate.sleep()

    def _run_manual(self, dt):
        """键盘积分位置，发布 /position_cmd。LOCK 状态下忽略移动轴。"""
        a = self.smooth_alpha
        for i in range(8):
            self.smooth_axes[i] = a * self._dz(self.joy_axes[i]) + (1.0 - a) * self.smooth_axes[i]

        # LOCK 状态：仅保持当前目标，不接受键盘移动
        if self._is_locked:
            vx = vy = vz = yaw_dot = 0.0
        else:
            # 偏航积分
            yaw_dot = -self.smooth_axes[CH_YAW] * self.omega
            self.target_yaw += yaw_dot * dt

            # 机体系速度 → 世界系
            fwd  =  self.smooth_axes[CH_FORWARD] * self.vel
            left =  self.smooth_axes[CH_LEFT]    * self.vel
            up   =  self.smooth_axes[CH_UP]      * self.vel
            # When yaw input is idle, keep translation aligned with the stable
            # target heading; while actively yawing, follow the actual heading.
            heading_yaw = self.drone_yaw if abs(yaw_dot) > 1e-3 else self.target_yaw
            cy, sy = np.cos(heading_yaw), np.sin(heading_yaw)
            vx = fwd * cy - left * sy
            vy = fwd * sy + left * cy
            vz = up

            self.target_pos[0] += vx * dt
            self.target_pos[1] += vy * dt
            self.target_pos[2] = max(0.2, self.target_pos[2] + vz * dt)

        # 手臂关节
        arm_delta = np.array([
            self._dz(self.joy_axes[CH_PITCH])  * self.arm_speed * dt,
            self._dz(self.joy_axes[CH_PITCH2]) * self.arm_speed * dt,
            self._dz(self.joy_axes[CH_ROLL])   * self.arm_speed * dt,
        ])
        self.arm_target = np.clip(self.arm_target + arm_delta, self.ARM_LO, self.ARM_HI)

        cmd = PositionCommand()
        cmd.header.stamp    = rospy.Time.now()
        cmd.header.frame_id = 'world'
        cmd.position.x  = float(self.target_pos[0])
        cmd.position.y  = float(self.target_pos[1])
        cmd.position.z  = float(self.target_pos[2])
        cmd.velocity.x  = float(vx)
        cmd.velocity.y  = float(vy)
        cmd.velocity.z  = float(vz)
        cmd.yaw         = float(self.target_yaw)
        cmd.yaw_dot     = float(yaw_dot)
        cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
        cmd.theta.position  = self.arm_target.tolist()
        cmd.theta.velocity  = [0.0, 0.0, 0.0]
        cmd.grab_cmd        = 2  # no-op: 不干预夹爪

        self.cmd_pub.publish(cmd)

    def _goal_picked_cb(self, msg):
        """双击目标点：仅保存规划器目标位置，不改变无人机 hover target_pos。
        （对齐 airgrasp：stage2 无人机保持当前悬停，只调幽灵臂；CONFIRM 后规划器才飞过去）
        注意：/goal_preview 以 5Hz 持续发布，_arm_hold 只在新目标出现时更新一次。
        """
        new_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        new_yaw = float(np.arctan2(siny, cosy))

        is_new_goal = (self._last_goal_pos is None or
                       np.linalg.norm(new_pos - self._last_goal_pos) > 0.05)
        if is_new_goal:
            self._arm_hold         = self.arm_target.copy()  # 仅新目标时锁定臂角
            self._goal_picked      = True
            self._mission_active   = False
            self._last_goal_pos    = new_pos.copy()
            # 保存规划器目标（与 hover target_pos 分开，不让无人机在 stage2 飞过去）
            self._planner_goal_pos = new_pos.copy()
            self._planner_goal_yaw = new_yaw
            rospy.logwarn('[DBG goal_picked] NEW GOAL pos=(%.2f,%.2f,%.2f) yaw=%.2f arm_hold=%s',
                new_pos[0], new_pos[1], new_pos[2], new_yaw,
                [round(v, 3) for v in self._arm_hold])

    def _run_planning(self):
        """透传规划器命令到 /position_cmd。
        阶段1 (未双击): 键盘调整 arm_target → 注入 hover cmd（设置起始臂角）
        阶段2 (已双击,未CONFIRM): 键盘调整 arm_target（仅改 ghost/final），
                                   hover cmd 锁定在 _arm_hold（保持实际臂不动→保证 init_thetas）
        阶段3 (已CONFIRM): 直接透传规划器 WITHYAWANDTHETA 轨迹 theta
        """
        dt = 1.0 / self.rate_hz
        arm_delta = np.array([
            self._dz(self.joy_axes[CH_PITCH])  * self.arm_speed * dt,
            self._dz(self.joy_axes[CH_PITCH2]) * self.arm_speed * dt,
            self._dz(self.joy_axes[CH_ROLL])   * self.arm_speed * dt,
        ])

        # ── arm_target 必须在 planner 检查之前更新 ────────────────────────
        # 无论规划器是否在运行，幽灵臂（ghost arm）都需要跟随键盘调整。
        # 如果放在 cmd is None 之后，规划器未启动时 arm_target 永不更新 → ghost 冻结。
        if not self._mission_active:
            self.arm_target = np.clip(self.arm_target + arm_delta, self.ARM_LO, self.ARM_HI)

        # 原子性地快照引用：防止 _planner_cb（在 ROS 回调线程）在本函数执行期间
        # 替换 self._planner_cmd，导致 theta 注入与 publish 之间发布了未修改的新消息。
        cmd = self._planner_cmd
        if cmd is None:
            # 规划器未启动/未连接：发布静态悬停命令维持当前位置和臂角
            if self.target_pos is not None:
                hover = PositionCommand()
                hover.header.stamp    = rospy.Time.now()
                hover.header.frame_id = 'world'
                hover.position.x = float(self.target_pos[0])
                hover.position.y = float(self.target_pos[1])
                hover.position.z = float(self.target_pos[2])
                hover.yaw         = float(self.target_yaw)
                hover.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
                arm_to_cmd = self._arm_hold if self._goal_picked else self.arm_target
                hover.theta.position = arm_to_cmd.tolist()
                hover.theta.velocity = [0.0, 0.0, 0.0]
                hover.grab_cmd       = 2  # no-op
                self.cmd_pub.publish(hover)
            return

        cmd.header.stamp = rospy.Time.now()

        # 切换到 PLANNING 后 2s 过渡期：覆盖 planner 位置为当前真实位置，防止掉高
        # （规划器 hover_p_ 固定在起飞时高度，切回后会拉回原高度）
        if self._mode_switch_time is not None:
            elapsed = (rospy.Time.now() - self._mode_switch_time).to_sec()
            if elapsed < 2.0:
                cmd.position.x = float(self.drone_pos[0])
                cmd.position.y = float(self.drone_pos[1])
                cmd.position.z = float(self.drone_pos[2])
                cmd.velocity.x = 0.0
                cmd.velocity.y = 0.0
                cmd.velocity.z = 0.0
            else:
                self._mode_switch_time = None  # 过渡期结束，恢复正常

        if self._mission_active:
            # 阶段3: 规划器已接管，直接透传 WITHYAWANDTHETA 轨迹
            stage = 3
        elif self._goal_picked:
            # 阶段2: 双击后 CONFIRM 前 — 键盘调整目标臂角（ghost），实际臂保持 _arm_hold
            cmd.theta.position = self._arm_hold.tolist()  # 实际臂不动
            cmd.theta.velocity = [0.0, 0.0, 0.0]
            stage = 2
        else:
            # 阶段1: 未选目标 — 键盘直接控制实际臂（设置起始臂角）
            cmd.theta.position = self.arm_target.tolist()
            cmd.theta.velocity = [0.0, 0.0, 0.0]
            stage = 1

        self._dbg_cnt += 1
        if self._dbg_cnt % 100 == 0:
            pub_theta = list(cmd.theta.position) if cmd.theta.position else []
            rospy.logwarn('[DBG planning stage=%d] arm_target=%s arm_hold=%s pub_theta=%s flag=%d',
                stage,
                [round(v, 3) for v in self.arm_target],
                [round(v, 3) for v in self._arm_hold],
                [round(v, 3) for v in pub_theta],
                cmd.trajectory_flag)

        self.cmd_pub.publish(cmd)


if __name__ == '__main__':
    try:
        node = JoyToPositionCmd()
        node.run()
    except rospy.ROSInterruptException:
        pass
