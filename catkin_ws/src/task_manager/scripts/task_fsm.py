#!/usr/bin/env python3
"""
task_fsm.py
===========
任务状态机节点：根据遮挡分析结果自动排序子任务。

状态序列：
  IDLE
   → SCAN         (等待感知数据)
   → CHECK_OCC    (检查遮挡)
        ├─ clear   → NAVIGATE_TGT → GRASP_TGT → DONE
        └─ occluded→ NAVIGATE_OBS → GRASP_OBS → CARRY → PLACE → CHECK_OCC

接口：
  订阅  /task/occlusion_state        std_msgs/String
        /task/obstacle_poses         geometry_msgs/PoseArray
        /task/obstacle_remove_poses  geometry_msgs/PoseArray
        /task/target_pose            geometry_msgs/PoseStamped
        /odom                        nav_msgs/Odometry
  发布  /uam_state_goal_cmd          quadrotor_msgs/UAMFullState  (含臂关节角)
        /arm_grab_cmd                std_msgs/Bool  (True=抓取 / False=释放)

臂关节角说明（弧度）：
  nav_arm_theta   = [0, 0, 0]         飞行中臂收起
  grasp_arm_theta = [π/2, 0, 0]       臂竖直朝下，准备抓取
  下降阶段使用 grasp_arm_theta，规划器同步优化飞行轨迹与臂轨迹

启动：将 ROS param ~auto_start 设为 true 则节点启动后自动从 IDLE 进入 SCAN；
      否则等待话题 /task/start (std_msgs/Bool, data=true) 触发。
"""

import os
import math
import rospy
import yaml
from enum import Enum
from threading import Lock

from std_msgs.msg      import String, Bool
from nav_msgs.msg      import Odometry
from geometry_msgs.msg import PoseStamped, PoseArray, Quaternion, Pose, Point
from quadrotor_msgs.msg import UAMFullState


# ─────────────────────────────────────────────────────────────────────────────
class State(Enum):
    IDLE         = 'IDLE'
    SCAN         = 'SCAN'
    CHECK_OCC    = 'CHECK_OCC'
    NAVIGATE_OBS = 'NAVIGATE_OBS'
    GRASP_OBS    = 'GRASP_OBS'
    CARRY        = 'CARRY'
    PLACE        = 'PLACE'
    NAVIGATE_TGT = 'NAVIGATE_TGT'
    GRASP_TGT    = 'GRASP_TGT'
    DONE         = 'DONE'


def _yaw_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2)
    q.z = math.sin(yaw / 2)
    q.x = q.y = 0.0
    return q


def _dist_xy(pos, goal) -> float:
    return math.hypot(pos[0] - goal[0], pos[1] - goal[1])


# ─────────────────────────────────────────────────────────────────────────────
class TaskFSM:

    def __init__(self):
        rospy.init_node('task_fsm', anonymous=False)

        config_path = rospy.get_param('~config_path',
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'cfg', 'config.yaml'))
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        fsm_cfg  = cfg.get('fsm', {})
        out_cfg  = cfg.get('output', {})

        self._hover_h      = float(fsm_cfg.get('hover_height', 0.5))
        self._place_dist   = float(fsm_cfg.get('place_dist', 0.6))
        self._place_z_off  = float(fsm_cfg.get('place_z_offset', 0.0))
        self._nav_xy_tol   = float(fsm_cfg.get('nav_xy_tol', 0.08))
        self._nav_z_tol    = float(fsm_cfg.get('nav_z_tol', 0.06))
        self._descend      = float(fsm_cfg.get('grasp_descend_dist', 0.25))
        self._carry_h      = float(fsm_cfg.get('carry_height', 0.8))
        self._goal_timeout = float(fsm_cfg.get('goal_timeout', 15.0))
        self._rescan       = bool(fsm_cfg.get('rescan_after_place', True))

        # 机械臂关节角配置
        self._nav_theta   = list(fsm_cfg.get('nav_arm_theta',   [0.0, 0.0, 0.0]))
        self._grasp_theta = list(fsm_cfg.get('grasp_arm_theta', [1.5708, 0.0, 0.0]))

        goal_topic   = out_cfg.get('goal_topic',     '/uam_state_goal_cmd')
        arm_topic    = out_cfg.get('arm_cmd_topic',  '/arm_grab_cmd')
        state_topic  = out_cfg.get('occlusion_state_topic', '/task/occlusion_state')
        tpose_topic  = out_cfg.get('target_pose_topic',     '/task/target_pose')
        oposes_topic = out_cfg.get('obstacle_poses_topic',  '/task/obstacle_poses')
        rposes_topic = out_cfg.get('remove_poses_topic',    '/task/obstacle_remove_poses')

        self._auto_start = rospy.get_param('~auto_start', False)

        # ── 状态缓存 ─────────────────────────────────────────────────────────
        self._lock          = Lock()
        self._state         = State.IDLE
        self._pos           = [0.0, 0.0, 0.0]   # current drone xyz
        self._occ_state     = 'unknown'
        self._tgt_pose      = None    # [x,y,z]
        self._obs_poses     = []      # list of [x,y,z]
        self._remove_poses  = []      # list of [x,y,z]
        self._current_obs_idx = 0     # which obstacle being handled
        self._nav_goal      = None    # active nav goal [x,y,z]
        self._nav_t0        = None    # navigation start time
        self._state_t0      = None    # state entry time

        # ── 发布器 ───────────────────────────────────────────────────────────
        self._pub_goal = rospy.Publisher(goal_topic, UAMFullState, queue_size=1)
        self._pub_arm  = rospy.Publisher(arm_topic,  Bool,         queue_size=1)
        self._pub_fsm_state = rospy.Publisher('/task/fsm_state', String,
                                              queue_size=1, latch=True)

        # ── 订阅器 ───────────────────────────────────────────────────────────
        rospy.Subscriber('/odom', Odometry, self._cb_odom, queue_size=1)
        rospy.Subscriber(state_topic,  String,      self._cb_occ,    queue_size=1)
        rospy.Subscriber(tpose_topic,  PoseStamped, self._cb_tpose,  queue_size=1)
        rospy.Subscriber(oposes_topic, PoseArray,   self._cb_oposes, queue_size=1)
        rospy.Subscriber(rposes_topic, PoseArray,   self._cb_rposes, queue_size=1)
        rospy.Subscriber('/task/start', Bool, self._cb_start, queue_size=1)

        rospy.Timer(rospy.Duration(0.1), self._step)
        rospy.loginfo('[task_fsm] Ready  auto_start=%s', self._auto_start)
        rospy.loginfo('[task_fsm] nav_arm_theta=%s  grasp_arm_theta=%s',
                      self._nav_theta, self._grasp_theta)

        if self._auto_start:
            rospy.sleep(1.0)
            self._enter(State.SCAN)

    # ── 订阅回调 ─────────────────────────────────────────────────────────────
    def _cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        with self._lock:
            self._pos = [p.x, p.y, p.z]

    def _cb_occ(self, msg: String):
        with self._lock:
            self._occ_state = msg.data

    def _cb_tpose(self, msg: PoseStamped):
        p = msg.pose.position
        with self._lock:
            self._tgt_pose = [p.x, p.y, p.z]

    def _cb_oposes(self, msg: PoseArray):
        with self._lock:
            self._obs_poses = [[p.position.x, p.position.y, p.position.z]
                               for p in msg.poses]

    def _cb_rposes(self, msg: PoseArray):
        with self._lock:
            self._remove_poses = [[p.position.x, p.position.y, p.position.z]
                                  for p in msg.poses]

    def _cb_start(self, msg: Bool):
        if msg.data and self._state == State.IDLE:
            self._enter(State.SCAN)

    # ── 状态机主循环 ─────────────────────────────────────────────────────────
    def _step(self, _event):
        with self._lock:
            state     = self._state
            pos       = list(self._pos)
            occ       = self._occ_state
            tgt       = list(self._tgt_pose) if self._tgt_pose else None
            obs_list  = list(self._obs_poses)
            rem_list  = list(self._remove_poses)
            obs_idx   = self._current_obs_idx

        if state == State.IDLE:
            return

        elif state == State.SCAN:
            # 等待目标点云就绪
            if tgt is not None and occ != 'unknown':
                self._enter(State.CHECK_OCC)

        elif state == State.CHECK_OCC:
            if occ == 'clear':
                rospy.loginfo('[task_fsm] Path clear → navigate to target')
                self._enter(State.NAVIGATE_TGT)
            elif occ == 'occluded':
                if obs_idx < len(obs_list):
                    rospy.loginfo(f'[task_fsm] Occluded by {len(obs_list)} obstacle(s) '
                                  f'→ handling obs #{obs_idx}')
                    self._enter(State.NAVIGATE_OBS)
                else:
                    rospy.logwarn('[task_fsm] No obstacle pose data — re-scanning')
                    self._enter(State.SCAN)

        elif state == State.NAVIGATE_OBS:
            if obs_idx >= len(obs_list):
                self._enter(State.CHECK_OCC); return
            goal_xy = obs_list[obs_idx][:2]
            hover_z = obs_list[obs_idx][2] + self._hover_h
            # 飞行阶段：臂收起，到达悬停点后换抓取构型并下降
            if not self._arrived(pos, [goal_xy[0], goal_xy[1], hover_z]):
                self._publish_goal(goal_xy[0], goal_xy[1], hover_z,
                                   arm_theta=self._nav_theta)
            else:
                low_z = obs_list[obs_idx][2] + self._hover_h - self._descend
                # 下降阶段：臂伸出朝下
                self._publish_goal(goal_xy[0], goal_xy[1], low_z,
                                   arm_theta=self._grasp_theta)
                if self._arrived(pos, [goal_xy[0], goal_xy[1], low_z]):
                    self._enter(State.GRASP_OBS)
            if self._nav_timeout():
                rospy.logwarn('[task_fsm] NAVIGATE_OBS timeout → retry scan')
                self._enter(State.SCAN)

        elif state == State.GRASP_OBS:
            self._pub_arm.publish(Bool(data=True))
            rospy.sleep(1.0)   # 等待夹爪关闭
            self._enter(State.CARRY)

        elif state == State.CARRY:
            if obs_idx >= len(rem_list):
                rospy.logwarn('[task_fsm] No remove pose → clear gripper')
                self._pub_arm.publish(Bool(data=False))
                self._enter(State.SCAN); return
            # 先爬升到 carry_height，再飞到清除点（臂收起避免碰撞）
            goal_x, goal_y = rem_list[obs_idx][:2]
            self._publish_goal(pos[0], pos[1], self._carry_h,
                               arm_theta=self._nav_theta)
            if abs(pos[2] - self._carry_h) < self._nav_z_tol:
                self._publish_goal(goal_x, goal_y, self._carry_h,
                                   arm_theta=self._nav_theta)
                if self._arrived(pos, [goal_x, goal_y, self._carry_h]):
                    self._enter(State.PLACE)
            if self._nav_timeout():
                rospy.logwarn('[task_fsm] CARRY timeout → release and rescan')
                self._pub_arm.publish(Bool(data=False))
                self._enter(State.SCAN)

        elif state == State.PLACE:
            if obs_idx < len(rem_list):
                rem_z = rem_list[obs_idx][2] + self._place_z_off + self._hover_h - self._descend
                # 放置下降阶段：臂伸出
                self._publish_goal(rem_list[obs_idx][0], rem_list[obs_idx][1], rem_z,
                                   arm_theta=self._grasp_theta)
                if self._arrived(pos, [rem_list[obs_idx][0], rem_list[obs_idx][1], rem_z]):
                    self._pub_arm.publish(Bool(data=False))
                    rospy.sleep(0.5)
                    with self._lock:
                        self._current_obs_idx += 1
                    if self._rescan:
                        self._enter(State.SCAN)
                    else:
                        self._enter(State.CHECK_OCC)
            if self._nav_timeout():
                self._pub_arm.publish(Bool(data=False))
                self._enter(State.SCAN)

        elif state == State.NAVIGATE_TGT:
            if tgt is None:
                self._enter(State.SCAN); return
            hover_z = tgt[2] + self._hover_h
            # 飞行阶段：臂收起
            if not self._arrived(pos, [tgt[0], tgt[1], hover_z]):
                self._publish_goal(tgt[0], tgt[1], hover_z,
                                   arm_theta=self._nav_theta)
            else:
                low_z = tgt[2] + self._hover_h - self._descend
                # 下降抓取阶段：臂伸出朝下，规划器同步规划臂轨迹
                self._publish_goal(tgt[0], tgt[1], low_z,
                                   arm_theta=self._grasp_theta)
                if self._arrived(pos, [tgt[0], tgt[1], low_z]):
                    self._enter(State.GRASP_TGT)
            if self._nav_timeout():
                rospy.logwarn('[task_fsm] NAVIGATE_TGT timeout → retry')
                self._enter(State.SCAN)

        elif state == State.GRASP_TGT:
            self._pub_arm.publish(Bool(data=True))
            rospy.sleep(1.5)
            self._enter(State.DONE)

        elif state == State.DONE:
            rospy.loginfo_once('[task_fsm] Task DONE — target grasped.')
            self._pub_fsm_state.publish(String(data='DONE'))

    # ── 辅助 ─────────────────────────────────────────────────────────────────
    def _enter(self, new_state: State):
        with self._lock:
            old = self._state
            self._state    = new_state
            self._state_t0 = rospy.Time.now()
            self._nav_t0   = rospy.Time.now()
        rospy.loginfo(f'[task_fsm] {old.value} → {new_state.value}')
        self._pub_fsm_state.publish(String(data=new_state.value))

    def _publish_goal(self, x, y, z, arm_theta=None):
        """发布 UAMFullState 目标，含机械臂关节角配置。"""
        if arm_theta is None:
            arm_theta = self._nav_theta

        msg = UAMFullState()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = 'world'

        msg.pose.position.x    = float(x)
        msg.pose.position.y    = float(y)
        msg.pose.position.z    = float(z)
        q = _yaw_quat(0.0)
        msg.pose.orientation.w = q.w
        msg.pose.orientation.x = q.x
        msg.pose.orientation.y = q.y
        msg.pose.orientation.z = q.z

        msg.theta  = [float(v) for v in arm_theta]
        msg.dtheta = [0.0] * len(arm_theta)   # 目标速度为零（静止构型）

        self._pub_goal.publish(msg)
        with self._lock:
            self._nav_goal = [x, y, z]

    def _arrived(self, pos, goal) -> bool:
        dxy = _dist_xy(pos, goal)
        dz  = abs(pos[2] - goal[2])
        return dxy < self._nav_xy_tol and dz < self._nav_z_tol

    def _nav_timeout(self) -> bool:
        with self._lock:
            t0 = self._nav_t0
        if t0 is None:
            return False
        return (rospy.Time.now() - t0).to_sec() > self._goal_timeout


if __name__ == '__main__':
    node = TaskFSM()
    rospy.spin()
